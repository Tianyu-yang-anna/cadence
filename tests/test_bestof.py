"""CPU-tiny tests for best-of-N reranking (utils/rerank.py).

Uses a random-init tiny GPT-2 (GPT2LMHeadModel(GPT2Config(...))) injected
into GPT2Scorer — no hub downloads — plus a word-level stub tokenizer.
"""
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from utils.rerank import GPT2Scorer, candidate_seed, fit_context, select_best

VOCAB = 100
MAX_LEN = 32


class StubTok:
    """Word-level stand-in for a HF tokenizer (deterministic, no downloads)."""
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        ids = [sum(ord(ch) for ch in w) % (VOCAB - 1) + 1 for w in text.split()]
        return {"input_ids": ids}


def tiny_scorer(batch_size=8):
    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=VOCAB, n_positions=MAX_LEN, n_embd=32,
                     n_layer=2, n_head=2)
    return GPT2Scorer("cpu", model=GPT2LMHeadModel(cfg), tokenizer=StubTok(),
                      batch_size=batch_size)


def manual_cont_nll(scorer, prompt, cont):
    """Hand computation: mean NLL over continuation positions only."""
    p, c = fit_context(scorer.tok(prompt)["input_ids"],
                       scorer.tok(cont)["input_ids"],
                       scorer.max_len, scorer.pad_id)
    ids = torch.tensor([p + c])
    with torch.no_grad():
        logits = scorer.model(input_ids=ids).logits
    logp = torch.log_softmax(logits[0, :-1].float(), -1)
    nll = -logp.gather(-1, ids[0, 1:, None]).squeeze(-1)
    return nll[len(p) - 1:].mean().item()  # token i predicted at position i-1


# ------------------------------------------------- (i) continuation-only NLL

def test_scores_continuation_only():
    s = tiny_scorer()
    prompt = "alpha beta gamma"
    c1, c2 = "delta epsilon zeta", "eta theta iota"
    got = s.score([prompt, prompt], [c1, c2])
    # matches the hand computation restricted to continuation positions
    assert abs(got[0] - manual_cont_nll(s, prompt, c1)) < 1e-4
    assert abs(got[1] - manual_cont_nll(s, prompt, c2)) < 1e-4
    # NLL moves when the continuation changes
    assert got[0] != got[1]
    # ... and does NOT equal the full-sequence mean (prompt positions
    # included), i.e. the scoring window excludes the prompt
    p = s.tok(prompt)["input_ids"]
    c = s.tok(c1)["input_ids"]
    ids = torch.tensor([p + c])
    with torch.no_grad():
        logits = s.model(input_ids=ids).logits
    logp = torch.log_softmax(logits[0, :-1].float(), -1)
    full = (-logp.gather(-1, ids[0, 1:, None]).squeeze(-1)).mean().item()
    assert abs(got[0] - full) > 1e-6


def test_padding_batch_invariance():
    # right-padding + attention mask: batched scores == solo scores
    s = tiny_scorer()
    prompts = ["alpha beta", "one two three four five six"]
    conts = ["gamma delta epsilon", "seven eight"]
    batch = s.score(prompts, conts)
    solo = [s.score([p], [c])[0] for p, c in zip(prompts, conts)]
    for a, b in zip(batch, solo):
        assert abs(a - b) < 1e-4


def test_empty_continuation_is_never_best():
    s = tiny_scorer()
    got = s.score(["alpha beta"], [""])
    assert got[0] == float("inf")


# --------------------------------------------------- (ii) left-truncation

def test_left_truncation_overlong_prompt():
    p_ids = list(range(1, 61))          # 60 prompt tokens > context 32
    c_ids = list(range(61, 71))         # 10 continuation tokens
    p_fit, c_fit = fit_context(p_ids, c_ids, MAX_LEN, fallback_id=0)
    assert c_fit == c_ids               # continuation intact
    assert len(p_fit) + len(c_fit) == MAX_LEN
    assert p_fit == p_ids[-22:]         # LEFT truncation keeps the suffix


def test_truncation_degenerate_cases():
    # continuation alone exceeds context: right-truncated, 1 prompt token kept
    p_fit, c_fit = fit_context([5, 6], list(range(1, 41)), MAX_LEN, 0)
    assert c_fit == list(range(1, MAX_LEN))
    assert p_fit == [6]
    # empty prompt gets the fallback conditioning token
    p_fit, c_fit = fit_context([], [1, 2, 3], MAX_LEN, 0)
    assert p_fit == [0] and c_fit == [1, 2, 3]
    # short pair untouched
    p_fit, c_fit = fit_context([1, 2], [3], MAX_LEN, 0)
    assert p_fit == [1, 2] and c_fit == [3]


def test_score_overlong_prompt_end_to_end():
    s = tiny_scorer()
    prompt = " ".join(f"w{i}" for i in range(50))  # 50 tokens > context 32
    cont = "final answer tokens here"
    got = s.score([prompt], [cont])[0]
    assert got != float("inf")
    assert abs(got - manual_cont_nll(s, prompt, cont)) < 1e-4


# ------------------------------------------- (iii) best-of argmin selection

class StubScorer:
    def __init__(self, table):
        self.table = table
        self.calls = []

    def score(self, prompts, continuations):
        self.calls.append((list(prompts), list(continuations)))
        return [self.table[c] for c in continuations]


def test_select_best_picks_argmin():
    table = {"a": 3.0, "d": 0.5, "b": 1.0, "e": 9.0, "c": 2.0, "f": 7.0}
    cand_texts = [["a", "d"], ["b", "e"], ["c", "f"]]  # N=3 candidates, B=2
    sc = StubScorer(table)
    best, idx = select_best(sc, ["p0", "p1"], cand_texts)
    assert best == ["b", "d"] and idx == [1, 0]
    # one flat scorer call, prompts aligned candidate-major
    ps, cs = sc.calls[0]
    assert ps == ["p0", "p1"] * 3
    assert cs == ["a", "d", "b", "e", "c", "f"]


def test_select_best_tie_keeps_first_candidate():
    best, idx = select_best(StubScorer({"x": 1.0, "y": 1.0}), ["p"],
                            [["x"], ["y"]])
    assert best == ["x"] and idx == [0]


# ------------------------------------------------ (iv) per-candidate seeds

def test_candidate_seed_deterministic_and_distinct():
    seeds = [candidate_seed(7, i) for i in range(16)]
    assert seeds == [7 * 10000 + i for i in range(16)]
    assert len(set(seeds)) == 16                       # distinct
    assert candidate_seed(7, 3) == candidate_seed(7, 3)  # deterministic
    assert candidate_seed(1, 0) != candidate_seed(0, 0)
