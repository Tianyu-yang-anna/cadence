# LOCAL ADDITION (not upstream): our repo root has a regular 'utils' package,
# and a regular package anywhere on sys.path beats a namespace package, so the
# vendored packages must be regular too — train_elf.py puts third_party/elf
# ahead of the repo root and these packages win there. See PROVENANCE.md.
