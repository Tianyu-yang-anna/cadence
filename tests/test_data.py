import numpy as np
import pytest
import torch

from data.prepare_tinystories import pack_per_doc
from data.prepare_wikitext import DOC_HEADING_RE, split_docs
from data.wikitext import (PaddedWindowDataset, SyntheticDataset, WindowBinDataset,
                           build_dataloader, build_dataset)


def test_split_docs_top_level_only():
    lines = [
        "",
        " = Doc One = \n",
        "",
        "some text a\n",
        " = = Section = = \n",          # sub-heading: must NOT start a new doc
        " = = = Subsub = = = \n",
        "more text\n",
        "",
        " = Doc Two = \n",
        "",
        "text c\n",
    ]
    docs = split_docs(lines)
    assert len(docs) == 2
    assert "Section" in docs[0] and "Doc One" in docs[0]
    assert "Doc Two" in docs[1]


def test_split_docs_ignores_inbody_lookalikes():
    # the train split contains ~970 body lines format-identical to headings
    # (wrapped stat legends, split equations); they have non-blank neighbors
    # and must NOT split the article
    lines = [
        "",
        " = Doc One = \n",
        "",
        "The abbreviations used in the table are :\n",
        " = Position ; GP = \n",        # lookalike, non-blank context
        " = Goals ; A = \n",            # lookalike, non-blank context
        "games played .\n",
        "",
        " = Doc Two = \n",
        "",
        "text c\n",
    ]
    docs = split_docs(lines)
    assert len(docs) == 2
    assert "Position ; GP" in docs[0]


def test_heading_regex():
    assert DOC_HEADING_RE.match(" = Valkyria Chronicles III = ")
    assert not DOC_HEADING_RE.match(" = = Gameplay = = ")
    assert not DOC_HEADING_RE.match("ordinary = text = here")


def test_window_bin_dataset(tmp_path):
    stream = np.arange(1000, dtype=np.uint16)
    path = tmp_path / "train.bin"
    stream.tofile(path)
    ds = WindowBinDataset(path, seq_len=64)
    assert len(ds) == 15  # final partial window dropped
    item = ds[1]
    assert item["input_ids"].dtype == torch.int64
    assert torch.equal(item["input_ids"], torch.arange(64, 128))
    assert torch.equal(item["labels"], item["input_ids"])
    ds_lim = WindowBinDataset(path, seq_len=64, limit_windows=4)
    assert len(ds_lim) == 4


def test_pack_per_doc_and_padded_dataset(tmp_path):
    doc_ids = [list(range(10)), list(range(100, 170))]
    windows, lengths = pack_per_doc(doc_ids, seq_len=64, pad_id=50256)
    assert windows.shape == (3, 64)          # 10 -> 1 window; 70 -> 2 windows
    assert lengths.tolist() == [10, 64, 6]
    assert windows[0, 10] == 50256           # padded tail

    windows.tofile(tmp_path / "windows_val.bin")
    np.save(tmp_path / "lengths_val.npy", lengths)
    ds = PaddedWindowDataset(tmp_path / "windows_val.bin",
                             tmp_path / "lengths_val.npy", seq_len=64)
    item = ds[0]
    assert item["attention_mask"][:10].sum() == 10
    assert item["attention_mask"][10:].sum() == 0
    assert (item["labels"][10:] == -100).all()
    assert (item["labels"][:10] == torch.arange(10)).all()


def test_synthetic_deterministic():
    a = SyntheticDataset(8, 32, 100, seed=5)
    b = SyntheticDataset(8, 32, 100, seed=5)
    c = SyntheticDataset(8, 32, 100, seed=6)
    assert torch.equal(a.ids, b.ids)
    assert not torch.equal(a.ids, c.ids)
    assert a.ids.max() < 100


def test_build_dataloader_synthetic(tiny_cfg):
    loader = build_dataloader(tiny_cfg, "train", batch_size=4, shuffle=True)
    batch = next(iter(loader))
    assert batch["input_ids"].shape == (4, tiny_cfg.model.seq_len)
    assert "attention_mask" not in batch
    val = build_dataset(tiny_cfg, "val")
    train = build_dataset(tiny_cfg, "train")
    assert not torch.equal(val.ids, train.ids)  # split-dependent seeds


def test_encode_docs_uses_eos():
    pytest.importorskip("transformers")
    from data.prepare_wikitext import encode_docs
    try:
        from transformers import GPT2TokenizerFast
        tok = GPT2TokenizerFast.from_pretrained("gpt2")
    except Exception as e:  # noqa: BLE001 - offline machine
        pytest.skip(f"gpt2 tokenizer unavailable: {e}")
    stream = encode_docs(["hello world", "second doc"], tok, tok.eos_token_id,
                         progress=False)
    assert stream.dtype == np.uint16
    assert (stream == tok.eos_token_id).sum() == 2
    assert stream[-1] == tok.eos_token_id
