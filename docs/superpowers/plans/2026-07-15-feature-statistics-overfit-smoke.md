# Feature Statistics and Unimodal Overfit Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible feature-quality statistics and a 16-sample overfit diagnostic for the text, audio, and vision unimodal classifiers, then run both on the complete EmotionTalk training cache.

**Architecture:** A focused `feature_statistics` module streams final feature shards and joins their sample IDs to the official manifest without loading the whole cache at once. A separate `overfit_smoke` module selects only rows where the requested modality is available, builds one deterministic training batch, and verifies that the existing `UnimodalClassifier` and training loop can memorize 16 examples. Both capabilities are exposed through the existing `bimer` CLI and write JSON artifacts.

**Tech Stack:** Python 3.11, NumPy, PyTorch 2.x, argparse, pytest.

## Global Constraints

- Keep the official EmotionTalk train/validation/test split; do not create a random validation split from train.
- Treat the 16-sample run only as a training-chain diagnostic, never as a reported baseline metric.
- Validate text/audio/vision dimensions as 768/1024/512 and reject non-finite values.
- Use the fixed seven-label order `neutral, joy, sadness, anger, surprise, fear, disgust`.
- Preserve all existing data-pipeline changes and do not overwrite cached feature shards.

---

### Task 1: Streaming feature-quality report

**Files:**
- Create: `src/bimer/feature_statistics.py`
- Create: `tests/test_feature_statistics.py`
- Modify: `src/bimer/cli.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: `read_manifest(path)`, `FeatureStore.paths(dataset, split)`, `FeatureStore.read(path)`, and `MODALITY_DIMS`.
- Produces: `compute_feature_statistics(records, store, dataset, split) -> dict[str, object]` and `write_feature_statistics(report, output_path) -> Path`.

- [ ] **Step 1: Write the failing statistics test**

```python
def test_feature_statistics_reports_labels_availability_and_dimensions(tmp_path):
    records, store = _fixture_with_two_rows_and_one_missing_vision(tmp_path)
    report = compute_feature_statistics(records, store, "emotiontalk", "train")
    assert report["sample_count"] == 2
    assert report["label_counts"] == {"neutral": 1, "joy": 1}
    assert report["modalities"]["vision"]["available_count"] == 1
    assert report["modalities"]["text"]["dimension"] == 4
    assert report["missing_manifest_samples"] == 0
```

- [ ] **Step 2: Run the test and confirm RED**

Run: `PYTHONPATH=src python3 -m pytest -q tests/test_feature_statistics.py`

Expected: collection fails because `bimer.feature_statistics` does not exist.

- [ ] **Step 3: Implement the minimal streaming report**

```python
def compute_feature_statistics(records, store, dataset, split):
    selected = [r for r in records if r.dataset == dataset and str(r.split) == split]
    manifest_by_id = {r.sample_id: r for r in selected}
    # Stream each shard, validate it with MODALITY_DIMS, and accumulate counts,
    # availability, L2 norms, zero-vector counts, and observed sample IDs.
    # Reject duplicate feature IDs and return missing/unexpected counts.
```

- [ ] **Step 4: Add the CLI contract test and implementation**

```bash
PYTHONPATH=src python3 -m bimer.cli feature-stats \
  --manifest manifest.jsonl --features features \
  --dataset emotiontalk --split train --output stats.json
```

The command must write UTF-8 JSON and print the same report.

- [ ] **Step 5: Run focused and full tests**

Run: `PYTHONPATH=src python3 -m pytest -q tests/test_feature_statistics.py tests/test_cli.py`

Expected: all tests pass.

### Task 2: Deterministic unimodal overfit diagnostic

**Files:**
- Create: `src/bimer/overfit_smoke.py`
- Create: `tests/test_overfit_smoke.py`
- Modify: `src/bimer/cli.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: official manifest records, final feature shards, `UnimodalClassifier`, `collate_dialogues`, `train_epoch`, and `evaluate_batches`.
- Produces: `run_unimodal_overfit_smoke(..., modalities=("text", "audio", "vision")) -> dict[str, object]` and a CLI JSON artifact.

- [ ] **Step 1: Write the failing sample-selection test**

```python
def test_overfit_example_uses_only_rows_where_modality_is_available(tmp_path):
    example = build_overfit_example(records, store, "vision", sample_count=2)
    assert len(example.sample_ids) == 2
    assert example.modality_mask[:, 2].all()
```

- [ ] **Step 2: Run the test and confirm RED**

Run: `PYTHONPATH=src python3 -m pytest -q tests/test_overfit_smoke.py`

Expected: collection fails because `bimer.overfit_smoke` does not exist.

- [ ] **Step 3: Implement selection and fitting**

```python
def run_unimodal_overfit_smoke(..., sample_count=16, max_epochs=200,
                               learning_rate=1e-2, target_accuracy=0.95):
    # Select the first deterministic available rows for each modality.
    # Train a dropout-free UnimodalClassifier on one repeated batch.
    # Stop once training accuracy reaches target_accuracy and loss is finite.
    # Return start/final loss, accuracy, epochs, selected IDs, and passed.
```

- [ ] **Step 4: Add the CLI contract**

```bash
PYTHONPATH=src python3 -m bimer.cli overfit-smoke \
  --manifest manifest.jsonl --features features \
  --dataset emotiontalk --split train --sample-count 16 \
  --max-epochs 200 --learning-rate 0.01 --device auto \
  --output overfit-smoke.json
```

The command exits non-zero if any requested modality does not reach the target.

- [ ] **Step 5: Run focused and full tests**

Run: `PYTHONPATH=src python3 -m pytest -q tests/test_overfit_smoke.py tests/test_cli.py`

Expected: all tests pass.

### Task 3: Run and archive the training-readiness evidence

**Files:**
- Create at runtime: `artifacts/analysis/emotiontalk-train-feature-statistics.json`
- Create at runtime: `artifacts/experiments/smoke/emotiontalk-unimodal-overfit.json`

**Interfaces:**
- Consumes: `artifacts/emotiontalk-train-00000-00964/merged-v1/bimer-output/emotiontalk.jsonl` and `artifacts/emotiontalk-train-00000-00964/merged-v1/features-emotiontalk-train-v4`.
- Produces: two JSON reports that establish feature quality and training-chain readiness.

- [ ] **Step 1: Generate the full training feature report**

Run the `feature-stats` command against EmotionTalk train and verify `sample_count=15413`, `shard_count=964`, no missing IDs, no unexpected IDs, and zero non-finite rows.

- [ ] **Step 2: Run the three 16-sample overfit diagnostics**

Run `overfit-smoke` for text, audio, and vision with seed 42 and require each modality to reach at least 0.95 training accuracy.

- [ ] **Step 3: Re-run the full repository test suite**

Run: `PYTHONPATH=src python3 -m pytest -q`

Expected: all tests pass with no warnings or errors.

- [ ] **Step 4: Record the formal-baseline boundary**

Do not run or report official validation/test metrics until EmotionTalk validation/test features and MELD train/dev/test features exist. The next cloud job must extract and verify those official splits without altering the manifest.
