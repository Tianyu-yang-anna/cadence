"""Gate tests for the vendored ELF port (third_party/elf + train_elf.py).

The vendored tree ships a package literally named `utils`, which collides with
the repo's own `utils` package; importing both in one process silently breaks
whichever loses. Every test that touches vendored code therefore runs in a
SUBPROCESS with third_party/elf first on sys.path — the same isolation
train_elf.py relies on — instead of importing it into the pytest process.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_isolated(body: str) -> str:
    """Run `body` in a fresh interpreter with the vendored tree first on
    sys.path. Prints from the body are the assertion channel."""
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT / 'third_party' / 'elf')!r})\n"
        f"sys.path.insert(1, {str(ROOT)!r})\n" + body
    )
    r = subprocess.run([sys.executable, "-c", script], capture_output=True,
                       text=True, timeout=600)
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    return r.stdout


def test_rotate_half_matches_einops_semantics():
    """The einops-free rotate_half must implement exactly
    '... (d r) -> ... d r' -> stack(-x2, x1) -> '... d r -> ... (d r)'."""
    run_isolated("""
import torch
from modules.layers import rotate_half
x = torch.randn(3, 5, 8)
got = rotate_half(x)
# hand-rolled einops semantics
ref = x.view(3, 5, 4, 2)
x1, x2 = ref[..., 0], ref[..., 1]
ref = torch.stack((-x2, x1), dim=-1).reshape(3, 5, 8)
assert torch.equal(got, ref), "rotate_half diverged from einops semantics"
print("ok")
""")


def test_two_pass_mask_visibility_semantics():
    """The two-pass replacement for the 3D pairwise encoder mask must
    reproduce its VISIBILITY semantics exactly:
      (a) cond rows see only cond -> their latents are invariant to any
          change in target tokens;
      (b) target rows see everything -> changing cond tokens moves them;
      (c) under label drop (the edited mask), target rows become invariant
          to cond tokens too.
    These properties pin the mask conversion without needing the
    transformers<4.45 native-3D path as a reference."""
    run_isolated("""
import torch
from transformers import T5EncoderModel, T5Config
from utils.encoder_utils import encode_text, build_self_attn_cond_masks
import numpy as np

torch.manual_seed(0)
cfg = T5Config(vocab_size=100, d_model=32, d_kv=8, d_ff=64,
               num_layers=2, num_heads=4)
hf = T5EncoderModel(cfg).eval()

class Enc:
    def __call__(self, input_ids=None, attention_mask=None, deterministic=True):
        return hf(input_ids=input_ids,
                  attention_mask=attention_mask).last_hidden_state
enc = Enc()

B, S, C = 2, 16, 6
ids = torch.randint(3, 100, (B, S))
is_cond = np.zeros((B, S), dtype=bool); is_cond[:, :C] = True
is_valid = np.ones((B, S), dtype=bool)
m3, attn, condm = build_self_attn_cond_masks(is_cond, is_valid, xp=np)
m3 = torch.from_numpy(m3)

def enc3(i):
    with torch.no_grad():
        return encode_text(input_ids=i, attention_mask=m3, encoder=enc,
                           latent_mean=0.0, latent_std=1.0, use_bf16=False)

base = enc3(ids)
ids_t = ids.clone(); ids_t[:, C:] = torch.randint(3, 100, (B, S - C))
moved_t = enc3(ids_t)
assert torch.allclose(base[:, :C], moved_t[:, :C], atol=1e-5), \
    "(a) cond rows saw the target"
assert not torch.allclose(base[:, C:], moved_t[:, C:], atol=1e-3), \
    "target change did not move target rows (degenerate test)"

ids_c = ids.clone(); ids_c[:, :C] = torch.randint(3, 100, (B, C))
moved_c = enc3(ids_c)
assert not torch.allclose(base[:, C:], moved_c[:, C:], atol=1e-3), \
    "(b) target rows blind to the condition"

# (c) label-drop edit: block (target row, cond col)
drop = torch.ones(B, 1, 1)
cm = torch.from_numpy(condm)
block = (1 - cm).unsqueeze(-1) * cm.unsqueeze(1)
m3d = m3 * (1 - drop * block)
def enc3d(i):
    with torch.no_grad():
        return encode_text(input_ids=i, attention_mask=m3d, encoder=enc,
                           latent_mean=0.0, latent_std=1.0, use_bf16=False)
b2, c2 = enc3d(ids), enc3d(ids_c)
assert torch.allclose(b2[:, C:], c2[:, C:], atol=1e-5), \
    "(c) dropped target rows still see the condition"
print("ok")
""")


def test_two_pass_equals_single_pass_when_all_rows_share_a_pattern():
    """With cond length 0 the pairwise mask has ONE row pattern (everything
    sees everything valid): the two-pass path must then agree with a direct
    2D call bit-for-bit — the cheapest exactness anchor available."""
    run_isolated("""
import torch, numpy as np
from transformers import T5EncoderModel, T5Config
from utils.encoder_utils import encode_text, build_self_attn_cond_masks
torch.manual_seed(1)
cfg = T5Config(vocab_size=100, d_model=32, d_kv=8, d_ff=64,
               num_layers=2, num_heads=4)
hf = T5EncoderModel(cfg).eval()
class Enc:
    def __call__(self, input_ids=None, attention_mask=None, deterministic=True):
        return hf(input_ids=input_ids,
                  attention_mask=attention_mask).last_hidden_state
enc = Enc()
B, S = 2, 12
ids = torch.randint(3, 100, (B, S))
is_cond = np.zeros((B, S), dtype=bool)
is_valid = np.ones((B, S), dtype=bool)
m3, _, _ = build_self_attn_cond_masks(is_cond, is_valid, xp=np)
with torch.no_grad():
    a = encode_text(input_ids=ids, attention_mask=torch.from_numpy(m3),
                    encoder=enc, latent_mean=0.0, latent_std=1.0,
                    use_bf16=False)
    b = encode_text(input_ids=ids, attention_mask=torch.ones(B, S),
                    encoder=enc, latent_mean=0.0, latent_std=1.0,
                    use_bf16=False)
assert torch.allclose(a, b, atol=1e-6), "two-pass != direct 2D on the trivial mask"
print("ok")
""")


def test_window_dataset_reassembles_and_covers_budget():
    """condition + input must concatenate back to the exact original window,
    the split must stay inside [30%, 70%], and the family budget arithmetic
    (7630 x 256 x 1024 = 2.0002B) must hold."""
    run_isolated("""
import numpy as np, torch
from pathlib import Path
import importlib.util
spec = importlib.util.spec_from_file_location(
    "train_elf", Path(%r) / "train_elf.py")
te = importlib.util.module_from_spec(spec); spec.loader.exec_module(te)
import tempfile, os
d = tempfile.mkdtemp()
rng = np.random.default_rng(0)
toks = rng.integers(3, 32000, size=8 * 1024, dtype=np.uint16)
toks.tofile(os.path.join(d, "t.bin"))
ds = te.WindowCondDataset(Path(d) / "t.bin", seed=7)
assert len(ds) == 8
for i in range(8):
    it = ds[i]
    w = np.concatenate([it["condition_input_ids"], it["input_ids"]])
    assert np.array_equal(w, toks[i * 1024:(i + 1) * 1024].astype(np.int64))
    frac = len(it["condition_input_ids"]) / 1024
    assert 0.29 <= frac <= 0.71, frac
assert 7630 * 256 * 1024 == 2000158720
print("ok")
""" % str(ROOT))
