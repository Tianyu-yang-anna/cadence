"""Prompted-continuation generation for the SEDD baseline (CADENCE patch).

SEDD (Score Entropy Discrete Diffusion, Lou et al. ICML 2024, arXiv 2310.16834)
is already a first-class algo in this vendored repo (configs/algo/sedd.yaml,
Diffusion._sedd_parameterization / _score_entropy / _analytic_update); it is
the kuleshov-group reimplementation the BD3-LM paper used as its own SEDD
baseline. Upstream only samples UNCONDITIONALLY (Diffusion._analytic_sampler),
so this file is the prompted twin of gen_prompted.py.

Why prompting is free here: SEDD's reverse process is absorbing-state, and
Diffusion._transp_transition() gives an already-revealed token i != mask_index
an outgoing edge only to itself. So in
    probs = staggered_score * transp_transition(x, dsigma)
every unmasked position has exactly ONE non-zero entry and is resampled to
itself with probability 1 (verified numerically). Seeding x with the prompt
therefore conditions generation exactly, with zero sampler surgery. The same
argument holds for _denoiser_update (it zeroes only the mask column).

Unlike BD3-LM/MDLM, SEDD has no semi-AR mode -- upstream states "sedd does not
support arbitrary-length sampling" (scripts/var_len/varlens_batch.sh). We
therefore denoise ONE fixed model.length window: [prompt][mask ... mask].

NFE knob: exactly `num_steps` analytic updates + 1 final denoiser update, i.e.
NFE = num_steps + 1, independent of sequence length. Drive it with
+prompted.num_steps=<T> (or algo.T=<T>). SEDD's OWT numbers use T=1024.

Emits the CADENCE benchmark JSONL rows {prompt, reference, generated} consumed
by eval_generation.py. Worker sharding: --shard i --nshards k -> rows i::k.

Usage (hydra config tree of main.py, plus +prompted.* keys):
  python gen_prompted_sedd.py algo=sedd model=small block_size=1024 \
      model.length=1024 model.attn_backend=sdpa \
      data=openwebtext-split "data.train=binwindows:..." \
      eval.checkpoint_path=<ckpt> \
      +prompted.benchmark=<bench.jsonl> +prompted.out=<gens.jsonl> \
      +prompted.n=1000 +prompted.num_steps=1024 \
      +prompted.shard=0 +prompted.nshards=8
"""
import json
import sys
from pathlib import Path

import hydra
import lightning as L
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataloader  # noqa: E402
import main as bd3_main  # noqa: E402


@torch.no_grad()
def prompted_analytic(model, prompt_ids: torch.Tensor, gen_tokens: int,
                      num_steps: int, eps: float = 1e-5, t_start: float = 0.0):
    """Seed a single fixed-length window with the prompt and denoise the rest.

    prompt_ids: [P] long on model.device. Returns (generated ids [G], nfe).

    t_start <= 0 means "auto": start the reverse trajectory at the time whose
    marginal mask rate equals the ACTUAL mask rate of the seeded window,
    (win - P)/win. Under the loglinear schedule move_chance(t) == t, so this
    is exact. Starting at t=1 instead (upstream's unconditional default) tells
    the model "everything is masked" while it can plainly see a clean prompt,
    and burns most of the step budget on a state the model never saw in
    training. Pass +prompted.t_start=1.0 to reproduce the naive behaviour.
    """
    device = model.device
    win = model.config.model.length            # 1024: SEDD is fixed-length
    gen_tokens = max(1, min(gen_tokens, win - 16))
    p_ids = prompt_ids[-(win - gen_tokens):]   # keep the RIGHTMOST prompt part
    p = p_ids.shape[0]

    x = model._sample_prior(1, win).to(device)  # all mask_index
    x[0, :p] = p_ids
    # ignore_bos=True was used in training: position 0 is never masked there.
    # Here position 0 is a real prompt token, which matches that convention.

    if t_start <= 0:
        t_start = max((win - p) / win, 2 * eps)
    t_start = min(t_start, 1.0)
    timesteps = torch.linspace(t_start, eps, num_steps + 1, device=device)
    dt = (t_start - eps) / num_steps
    nfe = 0
    for i in range(num_steps):
        if model.mask_index not in x:
            break
        t = timesteps[i] * torch.ones(1, 1, device=device)
        x = model._analytic_update(x=x, t=t, dt=dt)
        nfe += 1
    # final denoising step: strips any residual mask tokens
    t = timesteps[-1] * torch.ones(1, 1, device=device)
    x = model._denoiser_update(x=x, t=t)
    nfe += 1
    return x[0, p:p + gen_tokens], nfe


@hydra.main(version_base=None, config_path='configs', config_name='config')
def run(config):
    L.seed_everything(int(config.prompted.get('seed', 0)))
    tokenizer = dataloader.get_tokenizer(config)
    model = bd3_main._load_from_checkpoint(config=config, tokenizer=tokenizer)
    if config.eval.disable_ema:
        model.ema = None
    model.eval()
    model.config.sampling.kv_cache = False      # not supported for sedd
    assert model.parameterization == 'sedd', \
        f'gen_prompted_sedd.py expects algo=sedd, got {model.parameterization}'

    bench = Path(config.prompted.benchmark)
    out = Path(config.prompted.out)
    n = int(config.prompted.get('n', 1000))
    shard = int(config.prompted.get('shard', 0))
    nshards = int(config.prompted.get('nshards', 1))
    num_steps = int(config.prompted.get('num_steps', config.algo.T) or 0)
    if num_steps <= 0:
        num_steps = 1024                        # SEDD's OWT default
    t_start = float(config.prompted.get('t_start', 0.0))
    print(f'sedd prompted gen: num_steps={num_steps} max_nfe={num_steps + 1} '
          f'window={config.model.length} t_start={t_start or "auto"}',
          flush=True)

    rows = [json.loads(l) for l in bench.read_text().splitlines()][:n]
    rows = rows[shard::nshards]

    out.parent.mkdir(parents=True, exist_ok=True)
    done = []
    for r in tqdm(rows, desc=f'shard{shard}'):
        p_ids = tokenizer(r['prompt'], add_special_tokens=False,
                          return_tensors='pt')['input_ids'][0].to(model.device)
        ref_words = len(r['reference'].split())
        need = int(ref_words * 1.5) + 32        # same budget rule as gen_prompted.py
        gen, nfe = prompted_analytic(model, p_ids, need, num_steps=num_steps,
                                     t_start=t_start)
        text = tokenizer.decode(gen.tolist(), skip_special_tokens=True)
        # truncate to the reference word count (protocol: length-matched)
        text = ' '.join(text.split()[:ref_words])
        done.append({'prompt': r['prompt'], 'reference': r['reference'],
                     'generated': text, 'nfe': nfe})
    with open(out, 'w') as f:
        for d in done:
            f.write(json.dumps(d) + '\n')
    print(f'wrote {len(done)} rows -> {out}', flush=True)


if __name__ == '__main__':
    run()
