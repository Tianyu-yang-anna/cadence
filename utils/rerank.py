"""Best-of-N generation reranking: score candidate continuations with a
frozen GPT-2 LM; keep the candidate with the lowest mean per-token NLL
(lower = better).

Used by generate.py --best_of N: the planner samples N code ladders per
prompt (mid-scale conditional entropy makes plan quality vary across
samples), each ladder is decoded to text, and GPT2Scorer ranks the
candidates against the prompt text — a pure inference-cost quality lift.
"""
from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F


def candidate_seed(base_seed: int, cand_idx: int) -> int:
    """Deterministic, distinct per-candidate generator seed for best-of-N."""
    return base_seed * 10000 + cand_idx


def fit_context(prompt_ids: list[int], cont_ids: list[int], max_len: int,
                fallback_id: int) -> tuple[list[int], list[int]]:
    """Fit prompt + continuation into max_len positions.

    The prompt is truncated from the LEFT (the suffix nearest the
    continuation is the conditioning that matters). At least one prompt
    token is always kept so every continuation token has a preceding
    position to be predicted from; an empty prompt gets [fallback_id].
    In the degenerate case where the continuation alone exceeds
    max_len - 1 it is truncated from the right.
    """
    if not prompt_ids:
        prompt_ids = [fallback_id]
    if len(cont_ids) > max_len - 1:
        cont_ids = cont_ids[: max_len - 1]
    keep = min(len(prompt_ids), max_len - len(cont_ids))
    return prompt_ids[len(prompt_ids) - keep:], cont_ids


def select_best(scorer, prompt_texts: list[str],
                cand_texts: list[list[str]]) -> tuple[list[str], list[int]]:
    """Pick the lowest-scoring (best) candidate per row.

    cand_texts: [N][B], candidate-major (cand_texts[c][j] = candidate c for
    row j); prompt_texts: [B]. All N*B pairs go to the scorer as one flat
    call (the scorer batches internally). Returns (best_texts [B],
    best_cand_idx [B]); ties keep the lowest candidate index.
    """
    n_rows = len(prompt_texts)
    flat_p: list[str] = []
    flat_c: list[str] = []
    for row in cand_texts:
        assert len(row) == n_rows, "cand_texts must be [N][B] candidate-major"
        flat_p.extend(prompt_texts)
        flat_c.extend(row)
    scores = scorer.score(flat_p, flat_c)
    best_texts, best_idx = [], []
    for j in range(n_rows):
        col = [scores[c * n_rows + j] for c in range(len(cand_texts))]
        c_best = min(range(len(col)), key=col.__getitem__)
        best_idx.append(c_best)
        best_texts.append(cand_texts[c_best][j])
    return best_texts, best_idx


class GPT2Scorer:
    """Frozen GPT-2 LM scoring continuations conditioned on their prompts.

    score(prompts, continuations) -> mean per-token NLL over CONTINUATION
    tokens only (prompt positions are conditioning, never scored). Lower is
    better. Prompt and continuation are tokenized separately (well-defined
    boundary), concatenated, and batched with right-padding + attention
    masks; overlong prompts are truncated from the left to fit the model
    context (n_positions, 1024 for GPT-2). bf16 autocast on cuda.

    `model` / `tokenizer` may be injected for tests (e.g. a tiny
    random-init GPT2LMHeadModel(GPT2Config(...))); otherwise `model_name`
    is loaded from the hub.
    """

    def __init__(self, device, model_name: str = "gpt2-large", *,
                 model=None, tokenizer=None, batch_size: int = 8):
        self.device = torch.device(device)
        if model is None:
            from transformers import GPT2LMHeadModel
            model = GPT2LMHeadModel.from_pretrained(model_name)
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.tok = tokenizer
        self.batch_size = batch_size
        self.max_len = int(getattr(self.model.config, "n_positions", 1024))
        pad = getattr(self.tok, "eos_token_id", None)
        self.pad_id = pad if pad is not None else 0

    def _autocast(self):
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    @torch.no_grad()
    def score(self, prompts: list[str], continuations: list[str]) -> list[float]:
        """Mean per-token NLL of each continuation given its prompt."""
        assert len(prompts) == len(continuations)
        pairs = []
        for p, c in zip(prompts, continuations):
            p_ids = self.tok(p, add_special_tokens=False)["input_ids"]
            c_ids = self.tok(c, add_special_tokens=False)["input_ids"]
            pairs.append(fit_context(p_ids, c_ids, self.max_len, self.pad_id))
        out: list[float] = []
        for s in range(0, len(pairs), self.batch_size):
            out.extend(self._score_batch(pairs[s:s + self.batch_size]))
        return out

    def _score_batch(self, pairs: list[tuple[list[int], list[int]]]) -> list[float]:
        B = len(pairs)
        L = max(len(p) + len(c) for p, c in pairs)
        input_ids = torch.full((B, L), self.pad_id, dtype=torch.long)
        attn = torch.zeros(B, L, dtype=torch.long)
        cont = torch.zeros(B, L, dtype=torch.bool)   # True = continuation token
        for i, (p, c) in enumerate(pairs):
            seq = p + c
            input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            attn[i, :len(seq)] = 1                   # right-padding
            cont[i, len(p):len(seq)] = True
        input_ids = input_ids.to(self.device)
        attn = attn.to(self.device)
        cont = cont.to(self.device)
        with self._autocast():
            logits = self.model(input_ids=input_ids, attention_mask=attn).logits
        # token i is predicted from position i-1: shift targets/mask by one
        logp = F.log_softmax(logits[:, :-1].float(), dim=-1)
        tok_nll = -logp.gather(-1, input_ids[:, 1:, None]).squeeze(-1)  # [B, L-1]
        mask = cont[:, 1:].float()   # fit_context keeps >=1 prompt token, so
        n_tok = mask.sum(-1)         # every continuation token is scoreable
        nll = (tok_nll * mask).sum(-1) / n_tok.clamp(min=1.0)
        # empty continuation: define as +inf so it is never selected
        nll = torch.where(n_tok > 0, nll, torch.full_like(nll, float("inf")))
        return nll.cpu().tolist()
