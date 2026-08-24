"""Sharded 40B-token pipeline (CPU, no network): merge_bins/merge_codes on
synthetic shard dirs, dump_codes --window_range slicing, and prepare_owt
shard behavior (--splits train, source routes) with fakes."""
import json
import sys

import numpy as np
import pytest
import torch

from data import prepare_owt
from data.dump_codes import slice_windows
from data.merge_bins import merge_bins
from data.merge_codes import merge_codes
from data.wikitext import SyntheticDataset
from models.text_vqvae import TextVQVAE
from utils.codes import dump_codes
from utils.config import ModelConfig, QuantizerConfig, TransformerConfig

SCALES = [1, 4, 16]
SEQ_LEN = 16
VOCAB = 97


# ---------------------------------------------------------------- merge_bins

def make_bin_shard(root, k, tokens, with_heldout=False):
    d = root / f"shard{k}"
    d.mkdir()
    np.asarray(tokens, dtype=np.uint16).tofile(d / "train.bin")
    meta = {"tokenizer": "gpt2", "vocab_size": 50257, "sep_id": 50256,
            "seq_len": 256, "packing": "contiguous", "source": f"fake[{k}]",
            "splits": {"train": {"n_docs": len(tokens), "n_tokens": len(tokens)}}}
    if with_heldout:
        for split, off in (("val", 3), ("test", 7)):
            arr = np.arange(off, off + 5, dtype=np.uint16)
            arr.tofile(d / f"{split}.bin")
            meta["splits"][split] = {"n_docs": 1, "n_tokens": int(arr.size)}
        meta["splits"] = {s: meta["splits"][s] for s in ("val", "test", "train")}
    (d / "meta.json").write_text(json.dumps(meta))
    return d


def test_merge_bins_byte_exact_concat(tmp_path):
    parts = [list(range(10)), list(range(100, 125)), list(range(500, 507))]
    shards = [make_bin_shard(tmp_path, k, t, with_heldout=(k == 0))
              for k, t in enumerate(parts)]
    out = tmp_path / "merged"
    meta = merge_bins(shards, out)

    merged = np.fromfile(out / "train.bin", dtype=np.uint16)
    assert np.array_equal(merged, np.asarray(sum(parts, []), dtype=np.uint16))
    n = sum(len(t) for t in parts)
    assert meta["splits"]["train"] == {"n_docs": n, "n_tokens": n}
    # val/test byte-copied from shard 0, counts carried over
    for split, off in (("val", 3), ("test", 7)):
        got = np.fromfile(out / f"{split}.bin", dtype=np.uint16)
        assert np.array_equal(got, np.arange(off, off + 5, dtype=np.uint16))
        assert meta["splits"][split] == {"n_docs": 1, "n_tokens": 5}
    assert meta["source"] == ["fake[0]", "fake[1]", "fake[2]"]
    on_disk = json.loads((out / "meta.json").read_text())
    assert on_disk == meta


def test_merge_bins_rejects_meta_mismatch(tmp_path):
    shards = [make_bin_shard(tmp_path, k, [k, k], with_heldout=(k == 0))
              for k in range(2)]
    bad = json.loads((shards[1] / "meta.json").read_text())
    bad["seq_len"] = 512
    (shards[1] / "meta.json").write_text(json.dumps(bad))
    with pytest.raises(AssertionError, match="seq_len"):
        merge_bins(shards, tmp_path / "merged")


# --------------------------------------------------------------- merge_codes

def make_codes_shard(root, k, n_rows, scales=SCALES, ckpt="ck.pt", step=7,
                     with_heldout=False, seed=0):
    d = root / f"shard{k}"
    d.mkdir()
    rng = np.random.default_rng(seed + k)
    width = sum(scales)
    rows = rng.integers(0, 8192, size=(n_rows, width), dtype=np.int16)
    np.save(d / "codes_train.npy", rows)
    meta = {"ckpt": ckpt, "step": step, "scales": scales,
            "window_range": [0, n_rows], "splits": {"train": n_rows}}
    if with_heldout:
        for split in ("val", "test"):
            arr = rng.integers(0, 8192, size=(3, width), dtype=np.int16)
            np.save(d / f"codes_{split}.npy", arr)
            meta["splits"][split] = 3
    (d / "codes_meta.json").write_text(json.dumps(meta))
    return d, rows


def test_merge_codes_row_concat(tmp_path):
    made = [make_codes_shard(tmp_path, k, n, with_heldout=(k == 0))
            for k, n in enumerate((5, 9, 2))]
    shards = [d for d, _ in made]
    out = tmp_path / "merged"
    meta = merge_codes(shards, out)

    merged = np.load(out / "codes_train.npy")
    assert merged.dtype == np.int16
    assert np.array_equal(merged, np.concatenate([r for _, r in made]))
    assert meta["splits"]["train"] == 16
    assert meta["scales"] == SCALES and meta["ckpt"] == "ck.pt"
    assert meta["window_range"] is None
    for split in ("val", "test"):
        assert np.array_equal(np.load(out / f"codes_{split}.npy"),
                              np.load(shards[0] / f"codes_{split}.npy"))
        assert meta["splits"][split] == 3
    assert json.loads((out / "codes_meta.json").read_text()) == meta


def test_merge_codes_rejects_shard_mismatch(tmp_path):
    d0, _ = make_codes_shard(tmp_path, 0, 4, with_heldout=True)
    d1, _ = make_codes_shard(tmp_path, 1, 4, scales=[1, 4, 8])
    with pytest.raises(AssertionError, match="scales mismatch"):
        merge_codes([d0, d1], tmp_path / "m1")
    d2, _ = make_codes_shard(tmp_path, 2, 4, ckpt="other.pt")
    with pytest.raises(AssertionError, match="ckpt mismatch"):
        merge_codes([d0, d2], tmp_path / "m2")


# --------------------------------------------- dump_codes --window_range

def tiny_model():
    torch.manual_seed(0)
    mcfg = ModelConfig(vocab_size=VOCAB, seq_len=SEQ_LEN, d_model=32, d_code=8,
                       encoder=TransformerConfig(num_layers=1, num_heads=2),
                       decoder=TransformerConfig(num_layers=1, num_heads=2))
    qcfg = QuantizerConfig(scales=SCALES, codebook_size=64)
    return TextVQVAE(mcfg, qcfg).eval()


def test_window_range_matches_full_dump_rows(tmp_path):
    model = tiny_model()
    ds = SyntheticDataset(12, SEQ_LEN, VOCAB, seed=3)
    device = torch.device("cpu")
    full_path = tmp_path / "codes_full.npy"
    dump_codes(model, ds, device, n_windows=len(ds), batch_size=5,
               out_path=full_path)

    sub = slice_windows(ds, (3, 9))
    assert len(sub) == 6
    part_path = tmp_path / "codes_part.npy"
    dump_codes(model, sub, device, n_windows=len(sub), batch_size=5,
               out_path=part_path)
    assert np.array_equal(np.load(part_path), np.load(full_path)[3:9])


def test_slice_windows_clamps_and_rejects_empty():
    ds = SyntheticDataset(12, SEQ_LEN, VOCAB, seed=3)
    assert slice_windows(ds, None) is ds
    assert len(slice_windows(ds, (0, 999))) == 12  # B clamped to split length
    assert len(slice_windows(ds, (10, 999))) == 2
    with pytest.raises(AssertionError, match="empty"):
        slice_windows(ds, (12, 20))


# ----------------------------------------------------- prepare_owt sharding

class FakeTokenizer:
    sep_token_id = None
    eos_token_id = 0

    def __len__(self):
        return 100

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [[(ord(c) % 90) + 1 for c in t] for t in texts]}


def run_prepare(monkeypatch, tmp_path, docs, extra_args):
    import transformers

    class FakeAuto:
        @staticmethod
        def from_pretrained(name, use_fast=True):
            return FakeTokenizer()

    monkeypatch.setattr(transformers, "AutoTokenizer", FakeAuto)
    monkeypatch.setattr(prepare_owt, "doc_stream",
                        lambda source, data_files_range="":
                        ("fake", iter({"text": t} for t in docs)))
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["prepare_owt.py", "--out", str(out),
                                      "--batch_docs", "2"] + extra_args)
    prepare_owt.main()
    return out, json.loads((out / "meta.json").read_text())


def test_prepare_splits_train_only(monkeypatch, tmp_path):
    docs = ["abc", "defg", "hi"]  # 4 + 5 + 3 tokens with the appended EOS
    out, meta = run_prepare(monkeypatch, tmp_path, docs,
                            ["--splits", "train", "--max_tokens", "1000"])
    assert (out / "train.bin").exists()
    assert not (out / "val.bin").exists() and not (out / "test.bin").exists()
    assert set(meta["splits"]) == {"train"}
    assert meta["splits"]["train"] == {"n_docs": 3, "n_tokens": 12}
    tok = FakeTokenizer()
    want = [t for d in docs for t in tok([d])["input_ids"][0] + [0]]
    assert np.fromfile(out / "train.bin", dtype=np.uint16).tolist() == want


def test_prepare_default_three_splits(monkeypatch, tmp_path):
    docs = [f"doc{i:02d}" for i in range(8)]  # 6 tokens each
    out, meta = run_prepare(monkeypatch, tmp_path, docs,
                            ["--val_tokens", "6", "--test_tokens", "6",
                             "--max_tokens", "1000"])
    for split in ("val", "test", "train"):
        assert (out / f"{split}.bin").exists()
    # held-out splits carved first, remainder into train (batch_docs=2)
    assert meta["splits"]["val"]["n_tokens"] >= 6
    assert meta["splits"]["test"]["n_tokens"] >= 6
    total = sum(s["n_tokens"] for s in meta["splits"].values())
    assert total == 8 * 6


def test_doc_stream_source_routes(monkeypatch):
    import datasets
    calls = []

    def fake_load_dataset(*a, **k):
        calls.append((a, k))
        return iter(())

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    name, _ = prepare_owt.doc_stream("allenai/c4:en")
    assert name == "allenai/c4:en"
    assert calls[-1] == (("allenai/c4", "en"), {"split": "train", "streaming": True})

    name, _ = prepare_owt.doc_stream("allenai/c4:en", "3:5")
    assert "files[3:5)" in name
    a, k = calls[-1]
    assert a == ("allenai/c4",)
    assert k == {"data_files": {"train": ["en/c4-train.00003-of-01024.json.gz",
                                          "en/c4-train.00004-of-01024.json.gz"]},
                 "split": "train", "streaming": True}

    with pytest.raises(AssertionError):
        prepare_owt.doc_stream("allenai/c4", "3:5")  # config required for range
