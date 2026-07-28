# BIMER V5 Whisper robustness protocol

V5 is a post-hoc exploratory study. It does not modify the confirmatory V2
result and cannot replace the V2 defense deployment in v1.1.0.

## Scope

V5 adds an identity-initialized `ASRConsistentTextAdapter` to the V2 fusion
architecture. The adapter consumes the 768-dimensional text feature and four
text-quality values, applies a 128-dimensional residual bottleneck, and returns
a 768-dimensional feature. Language embedding remains disabled because its V2
ablation was unsupported.

Training uses aligned human/Whisper views from the locked 10% training-dialogue
subset:

```text
L = L_clean_CE + 0.5 * L_whisper_CE + beta * JS(p_clean, p_whisper)
```

Only `beta=0.05` and `beta=0.10` are screened with seed 42. Audio, vision,
standard text features, official splits, labels, `sample_id`, and `context_id`
remain unchanged.

## Validation-only decision

A candidate passes only if all checks hold:

- clean bilingual weighted-F1 delta is at least -0.3 percentage points;
- Whisper bilingual weighted-F1 gain is at least 1.5 points;
- each dataset gains at least 0.5 points under Whisper text;
- clean bilingual macro-F1 delta is at least -0.3 points;
- 10 dB audio and 50% video-drop weighted-F1 deltas are each at least -0.5
  points.

Screen and formal commands always use `--skip-test`. A passing configuration is
atomically frozen before formal seeds 42, 123, and 2026. The guarded exploratory
test requires a formal-completion marker and consumes an immutable
`TEST_EVALUATED` marker on its first attempt, including a failed attempt.

If neither beta passes, V5 stops immediately and is retained as a negative
result. Thresholds must not be relaxed after observing validation output.

## Cost and artifacts

`scripts/run_v5_autodl.sh` enforces a cumulative 36,000-second GPU budget,
stage timeout, resumable result checks, status files, SHA-256 packaging, and
automatic shutdown. The local result archive must be hash-verified before any
cloud cleanup.

The public repository may contain code, configuration, aggregate reports, and
the final decision. Paired per-sample features, predictions, checkpoints, and
licensed media remain in the private evidence package.
