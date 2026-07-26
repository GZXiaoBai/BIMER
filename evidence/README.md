# BIMER evidence registry

This directory contains portable integrity manifests for private thesis
evidence. The referenced datasets, features, checkpoints, predictions, and
licensed media are intentionally excluded from the public repository.

Paths in `private-artifacts.sha256` are relative to the repository root. Verify
the local evidence bundle with:

```bash
python -m bimer.cli verify-evidence \
  --manifest evidence/private-artifacts.sha256 \
  --root .
```

The registry proves which local artifacts support the published aggregate
results; it does not grant permission to redistribute those artifacts.
