"""Benchmark generation for the ELF baseline (arXiv 2605.10938).

Runs ELF's OWN conditional recipe on our prefix-continuation benchmarks: the
prompt's clean embeddings are prepended and preserved (exactly the conditional
pathway the arm was trained with), the continuation region starts from noise
and follows the ODE/SDE rollout, and the final latent decodes through the DLM
head. Output rows are our standard {index, prompt, reference, generated} so
eval_generation.py scores them like every other system.

Protocol note: like every family member (generate.py run_benchmark), the
generated text is word-truncated to the reference length before scoring, so
ROUGE/MAUVE are not length-confounded.

Usage:
  python generate_elf.py --run_dir runs/elf_owt2_t5_pre \
      --benchmark data/benchmarks/wikipedia.jsonl --out gens.jsonl --n 1000 \
      --steps 64 --cfg 2.0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "third_party" / "elf"))

from train_elf import SEQ_LEN, build_encoder  # noqa: E402 (sets up the same path)
from configs.config import Config, SamplingConfig  # noqa: E402 (vendored)
from modules.model import ELF_models  # noqa: E402
from utils.data_utils import get_dataloader, load_jsonl_dataset  # noqa: E402
from utils.encoder_utils import encode_text  # noqa: E402
from utils.generation_utils import (_dlm_decode_batch,  # noqa: E402
                                    _generate_samples_single_batch,
                                    mask_after_eos, shift_left)
from utils.sampling_utils import get_sampling_steps  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True,
                    help="train_elf.py run dir holding ckpt_*.pt + latest.txt")
    ap.add_argument("--benchmark", required=True,
                    help="JSONL with {prompt, reference} rows")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--sc_cfg", type=float, default=1.0)
    ap.add_argument("--method", choices=["ode", "sde"], default="ode")
    ap.add_argument("--sde_gamma", type=float, default=0.0)
    ap.add_argument("--max_input", type=int, default=672,
                    help="cond truncation in T5 tokens (~60% of the window "
                         "in GPT-2 tokens, +7% T5 inflation)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--use_ema", default="true", choices=["true", "false"],
                    help="false = sample the raw weights. The paper's EMA "
                         "decay 0.9999 is calibrated for its 95k-step runs; "
                         "at the family's 7630 steps that EMA still holds "
                         "~47%% of the RANDOM INIT (0.9999^7630 = 0.466) and "
                         "decodes garbage — measured: raw weights reconstruct "
                         "a clean latent at 100%%, the miscalibrated EMA at "
                         "51%%. Runs whose training used a budget-rescaled "
                         "decay can sample the EMA again.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.run_dir)
    ck_name = (run_dir / "latest.txt").read_text().strip()
    payload = torch.load(run_dir / ck_name, map_location=device,
                         weights_only=False)
    run_cfg = json.loads((run_dir / "config.json").read_text())

    # the training-time recipe constants the rollout must reproduce
    config = Config()
    config.max_length = SEQ_LEN
    config.max_input_length = args.max_input
    config.pad_token = "eos"
    config.denoiser_p_mean, config.denoiser_p_std = -1.5, 0.8
    config.denoiser_noise_scale = 2.0
    config.t_eps = 0.05
    config.self_cond_prob = 0.5
    config.label_drop_prob = 0.1
    config.latent_mean = float(payload["latent_mean"])
    config.latent_std = float(payload["latent_std"])
    samp = SamplingConfig()
    samp.sampling_method = args.method
    samp.time_schedule = "logit_normal"
    samp.sde_gamma = args.sde_gamma

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("t5-small", use_fast=True)
    enc_cfg, encoder = build_encoder(payload["encoder_kind"], device,
                                     seed=int(run_cfg.get("seed", 0)))

    model = ELF_models["ELF-B"](
        text_encoder_dim=enc_cfg.d_model, max_length=SEQ_LEN,
        bottleneck_dim=128, num_time_tokens=4, num_self_cond_cfg_tokens=4,
        num_model_mode_tokens=4, vocab_size=len(tokenizer)).to(device)
    # the paper's eval samples the EMA; see --use_ema for when not to
    model.load_state_dict(payload["ema"] if args.use_ema == "true"
                          else payload["model"])
    model.eval()
    print(f"[generate_elf] {run_dir.name} step {payload['step']} "
          f"encoder={payload['encoder_kind']} latent=({config.latent_mean:.3f},"
          f"{config.latent_std:.3f}) steps={args.steps} cfg={args.cfg}",
          flush=True)

    rows = load_jsonl_dataset(args.benchmark, tokenizer,
                              input_key="prompt", output_key="reference")
    rows = rows[: args.n]
    loader = get_dataloader(
        rows, batch_size=args.batch, shuffle=False, num_workers=0,
        drop_last=False, max_seq_length=SEQ_LEN,
        pad_token_id=int(tokenizer.pad_token_id or 0),
        max_input_seq_length=args.max_input, distributed=False)

    generator = torch.Generator().manual_seed(args.seed)
    eos_id = tokenizer.eos_token_id
    pad_id = int(tokenizer.pad_token_id or 0)
    out_f = open(args.out, "w")
    done, t0 = 0, time.time()
    with torch.no_grad():
        for batch in loader:
            ids = torch.from_numpy(np.asarray(batch["input_ids"])).to(device).long()
            enc_mask = torch.from_numpy(
                np.asarray(batch["encoder_attention_mask"])).to(device).float()
            cond_mask = torch.from_numpy(
                np.asarray(batch["cond_seq_mask"])).to(device).float()
            bsz = ids.shape[0]
            t_steps = get_sampling_steps(
                n_steps=args.steps, time_schedule=samp.time_schedule,
                P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
                device=device, dtype=torch.float32)
            cond_seq = encode_text(
                input_ids=ids, attention_mask=enc_mask, encoder=encoder,
                latent_mean=config.latent_mean, latent_std=config.latent_std)
            z = (torch.randn((bsz, SEQ_LEN, enc_cfg.d_model),
                             generator=generator)
                 * config.denoiser_noise_scale).to(device)
            latent = _generate_samples_single_batch(
                model=model, generator=generator, z=z, t_steps=t_steps,
                cond_seq=cond_seq.float(), cond_seq_mask=cond_mask,
                config=config, sampling_config=samp,
                cfg_scale=args.cfg, self_cond_cfg_scale=args.sc_cfg)
            pred = _dlm_decode_batch(z=latent, model=model,
                                     t_final_val=t_steps[-1].item(),
                                     config=config,
                                     self_cond_cfg_scale=args.sc_cfg)
            cond_lens = cond_mask.to(torch.int32).sum(dim=1)
            pred = shift_left(pred, cond_lens, pad_id)
            pred = mask_after_eos(pred, eos_token_id=eos_id, pad_token_id=pad_id)
            for i in range(bsz):
                text = tokenizer.decode(pred[i].detach().cpu().numpy(),
                                        skip_special_tokens=True)
                # the family protocol (generate.py run_benchmark): generated
                # text is WORD-TRUNCATED to the reference length so
                # ROUGE/MAUVE are not length-confounded — without this ELF
                # rows came out ~1.9x the reference and were incomparable
                n_ref_words = len(str(batch["target"][i]).split())
                text = " ".join(text.split()[:n_ref_words])
                out_f.write(json.dumps({
                    "index": done, "prompt": batch["input"][i],
                    "reference": batch["target"][i],
                    "generated": text}) + "\n")
                done += 1
            print(f"[generate_elf] {done}/{len(rows)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    out_f.close()
    print(f"[generate_elf] wrote {args.out} ({done} rows)", flush=True)


if __name__ == "__main__":
    main()
