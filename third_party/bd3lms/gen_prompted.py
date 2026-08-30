"""Prompted-continuation generation for BD3-LM / MDLM baselines (CADENCE
patch; the upstream repo only samples unconditionally).

The sampler is a prompt-seeded variant of Diffusion._semi_ar_sampler:
x_accum starts as [EOS-pad][BOS][prompt] (left-padded so the prompt ends on a
block boundary — BD3's attention blocks are anchored at multiples of
block_size from position 0), then each stride appends a fully-masked block
and denoises it inside the repo's own sliding context window. Unmasked
(prompt/committed) tokens are never rewritten by the absorbing-state update,
so the prompt conditions generation exactly like previously generated blocks.

Emits the CADENCE benchmark JSONL rows {prompt, reference, generated}
consumed by eval_generation.py. Worker sharding: --shard i --nshards k
processes rows i::k (one process per GPU; merge = simple concat downstream).

Usage (hydra config tree of main.py, plus +prompted.* keys):
  python gen_prompted.py algo=bd3lm model=small block_size=16 \
      data=openwebtext-split "data.train=binwindows:..." \
      eval.checkpoint_path=<ckpt> \
      +prompted.benchmark=<bench.jsonl> +prompted.out=<gens.jsonl> \
      +prompted.n=1000 +prompted.seed=0 +prompted.shard=0 +prompted.nshards=1
"""
import json
import math
import os
import sys
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataloader  # noqa: E402
import main as bd3_main  # noqa: E402


@torch.no_grad()
def prompted_semi_ar(model, prompt_ids: torch.Tensor, gen_tokens: int,
                     num_steps: int, context_size: int = 1024):
    """prompt_ids: [P] long on model.device. Returns generated ids [G]."""
    device = model.device
    block = model.block_size
    mdlm_semi_ar = (model.config.algo.name == 'mdlm'
                    and model.config.model.length > model.block_size)
    stride = 512 if mdlm_semi_ar else block
    bos = model.tokenizer.bos_token_id
    eos = model.tokenizer.eos_token_id
    seeded = torch.cat([torch.tensor([bos], device=device), prompt_ids])
    # left-pad with EOS so the seeded region ends on a stride boundary
    # (in-distribution: EOS separates documents in training bins)
    pad = (-seeded.shape[0]) % stride
    seeded = torch.cat([torch.full((pad,), eos, device=device,
                                   dtype=torch.long), seeded])
    x_accum = seeded[None, :].clone()                       # [1, P0]
    p0 = x_accum.shape[1]
    n_strides = math.ceil(gen_tokens / stride)
    ones = torch.ones((1, 1), dtype=model.dtype, device=device)
    for _ in range(n_strides):
        x = model._sample_prior(1, stride).to(device)       # all-mask block
        x_accum = torch.cat((x_accum, x), dim=1)
        end_idx = x_accum.shape[1]
        start_idx = max(end_idx - context_size, 0)
        fwd_idx = torch.arange(start_idx, end_idx, device=device)
        dt = 1 / num_steps
        p_x0_cache = None
        timesteps = torch.linspace(1, 0, num_steps, device=device)
        t = 1
        for i in range(num_steps):
            if model.mask_index not in x_accum:
                break
            if model.config.sampling.first_hitting:
                u = np.random.rand()
                num_masked = (x_accum[:, fwd_idx]
                              == model.mask_index).sum(-1).item()
                if num_masked == 0:
                    break
                t *= u ** (1 / num_masked)
            else:
                t = timesteps[i]
            p_x0_cache, x_next = model._ddpm_caching_update(
                x=x_accum[:, fwd_idx], t=t * ones, dt=dt, p_x0=p_x0_cache)
            x_accum[:, fwd_idx] = x_next
    return x_accum[0, p0:p0 + gen_tokens]


@hydra.main(version_base=None, config_path='configs', config_name='config')
def run(config):
    L.seed_everything(int(config.prompted.get('seed', 0)))
    tokenizer = dataloader.get_tokenizer(config)
    model = bd3_main._load_from_checkpoint(config=config, tokenizer=tokenizer)
    if config.eval.disable_ema:
        model.ema = None
    model.eval()
    # KV cache off: prompt lengths are arbitrary, cache indexing assumes
    # stride-aligned growth from position 0
    model.config.sampling.kv_cache = False

    bench = Path(config.prompted.benchmark)
    out = Path(config.prompted.out)
    n = int(config.prompted.get('n', 1000))
    shard = int(config.prompted.get('shard', 0))
    nshards = int(config.prompted.get('nshards', 1))
    num_steps = int(config.prompted.get('num_steps', config.algo.T) or 0)
    if num_steps <= 0:
        # algo.T == 0 means continuous time in the upstream configs; the
        # denoise loop needs a finite step budget (first_hitting exits
        # early once the stride is fully unmasked)
        num_steps = 1024
    rows = [json.loads(l) for l in bench.read_text().splitlines()][:n]
    rows = rows[shard::nshards]

    out.parent.mkdir(parents=True, exist_ok=True)
    done = []
    for r in tqdm(rows, desc=f'shard{shard}'):
        p_ids = tokenizer(r['prompt'], add_special_tokens=False,
                          return_tensors='pt')['input_ids'][0].to(model.device)
        p_ids = p_ids[-768:]                     # leave >=256 in-window budget
        need = int(len(r['reference'].split()) * 1.5) + 32
        need = min(need, 1400)
        gen = prompted_semi_ar(model, p_ids, need,
                               num_steps=num_steps)
        text = tokenizer.decode(gen.tolist(), skip_special_tokens=True)
        # truncate to the reference word count (protocol: length-matched)
        words = text.split()
        text = ' '.join(words[:len(r['reference'].split())])
        done.append({'prompt': r['prompt'], 'reference': r['reference'],
                     'generated': text})
    with open(out, 'w') as f:
        for d in done:
            f.write(json.dumps(d) + '\n')
    print(f'wrote {len(done)} rows -> {out}', flush=True)


if __name__ == '__main__':
    run()
