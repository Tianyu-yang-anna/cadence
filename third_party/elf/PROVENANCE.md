Vendored from https://github.com/lillian039/ELF branch `pytorch_elf`
(commit at clone time 2026-09-04, MIT License — see LICENSE).

Paper: "ELF: Embedded Language Flows", arXiv:2605.10938.

Local modifications are kept OUT of this directory: our training wrapper is
/train_elf.py, the data adapter and encoder builders live in
/models/elf_adapter.py. Files here are unmodified upstream except where a
header comment says otherwise.

Local modifications inside this directory:
- utils/__init__.py, modules/__init__.py, configs/__init__.py added (empty
  markers): upstream ships namespace packages, which lose to our repo-root
  regular `utils` package no matter the sys.path order.
- modules/layers.py: the three einops calls (rotate_half x2, RoPE repeat)
  replaced with equivalent pure-torch reshapes — einops is absent from the
  cluster's packed env and adding it would force a global env version bump.
  Equivalence pinned by tests/test_elf_port.py::test_rotate_half_matches_einops.
