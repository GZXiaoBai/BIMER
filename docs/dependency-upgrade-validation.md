# Dependency Upgrade Validation

Date: 2026-07-26

## Scope

The engineering-hardening work upgraded `transformers` from 4.57.6 to 5.14.1
to remove published vulnerabilities reported by `pip-audit`. The V2 checkpoint,
encoder snapshots, preprocessing code, label order, and model architecture were
not changed.

## Feature parity check

The comparison used the pinned local encoder assets in
`artifacts/models/huggingface`, CPU inference, evaluation mode, and offline
loading.

| Encoder | Test input | Shape | Maximum absolute difference |
|---|---|---:|---:|
| XLM-RoBERTa-base | One Chinese and one English utterance | `2 × 768` | `0.0` |
| Wav2Vec2 XLS-R 300M | Two deterministic sine-wave clips | `2 × 1024` | `0.0` |

Both comparisons were also exactly equal under `numpy.allclose` with
`rtol=1e-5` and `atol=1e-6`.

## Security and regression evidence

- `pip-audit`: no known vulnerabilities after the upgrade.
- Full test suite: 319 tests passed before the focused coverage additions.
- Offline V2 runtime assembly completed using the pinned local assets.

This validation supports treating the dependency change as an engineering and
security update, not as a change to the reported research protocol.
