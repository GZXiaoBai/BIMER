# Contributing to BIMER

Thank you for helping improve BIMER. This public repository contains code,
configuration, tests, aggregate results and documentation only.

## Development setup

Use Python 3.11 and the locked environment:

```bash
uv sync --extra dev --extra inference --frozen
```

Before opening a pull request, run:

```bash
uv run python scripts/check_public_tree.py --root .
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run mypy \
  src/bimer/deployment.py \
  src/bimer/integrity.py \
  src/bimer/runtime.py \
  src/bimer/calibration.py
uv run pytest --cov=bimer --cov-fail-under=80
uv run pip-audit
```

## Repository boundaries

Do not commit:

- MELD or EmotionTalk media, annotations, derived per-sample records or cached
  features;
- model checkpoints, pretrained encoder caches or files larger than 10 MiB;
- private demonstration videos, browser traces, access tokens or credentials;
- local absolute paths, especially cloud instance paths.

Use synthetic fixtures in tests. Aggregate metrics may be contributed when they
do not permit reconstruction of restricted records.

## Change expectations

- Preserve the public `analyze_dialogue()` interface.
- Add regression tests for behavioral changes.
- Do not tune a model or threshold against an official test set.
- Separate confirmatory V2 results from exploratory V3 work.
- Report negative results and limitations without changing the evaluation
  protocol after observing results.

By contributing, you agree that your code contribution is licensed under
Apache-2.0. Dataset and model licenses remain independent.
