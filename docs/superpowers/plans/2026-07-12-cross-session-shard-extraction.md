# Cross-Session Global Shard Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add globally numbered shard ranges, strict feature-range verification, and a documented Kaggle restore workflow so EmotionTalk train can be completed safely across sessions.

**Architecture:** A focused `shard_ranges` module validates and slices official-split records. The existing parallel runner receives a global shard offset, while a separate verifier checks manifest IDs and feature arrays and publishes range-completion metadata only after success. CLI commands expose extraction and verification without changing existing full-run behavior.

**Tech Stack:** Python 3.11, NumPy 2.x, PyTorch 2.x, argparse, pathlib, pytest 8.x, Kaggle dual T4.

## Global Constraints

- Preserve official record order and official dataset splits.
- Range syntax is left-closed and right-open: `[start_shard, end_shard)`.
- Both range arguments are supplied together or both omitted.
- Ranged extraction requires parallel mode plus explicit dataset and split.
- Shard indices remain global across staging, final features, and completion metadata.
- Existing atomic writes, strict IDs, feature widths 768/1024/512, masks, and finite-value checks remain mandatory.
- EmotionTalk train uses shard size 16, 15,413 samples, and 964 shards numbered 0 through 963.
- Existing validation and test outputs are never overwritten.
- Never print or persist `HF_TOKEN` or raw EmotionTalk media.

---

### Task 1: Range Resolution and Global Record Shard Numbers

**Files:**
- Create: `src/bimer/shard_ranges.py`
- Modify: `src/bimer/parallel_feature_extraction.py`
- Create: `tests/test_shard_ranges.py`
- Modify: `tests/test_parallel_feature_extraction.py`

**Interfaces:**
- Produces: `ShardRange`, `resolve_shard_range`, `slice_shard_range`.
- Modifies: `record_shards(records, shard_size, shard_index_offset=0)` and `ParallelFeatureExtractionConfig.shard_index_offset`.

- [ ] **Step 1: Write failing range tests**

```python
def test_resolve_full_and_partial_shard_ranges():
    assert resolve_shard_range(15413, 16, None, None) == ShardRange(0, 964, 964)
    assert resolve_shard_range(15413, 16, 120, 240) == ShardRange(120, 240, 964)


@pytest.mark.parametrize("start,end", [(0, None), (None, 1), (-1, 1), (2, 2), (3, 2), (0, 965)])
def test_resolve_shard_range_rejects_invalid_bounds(start, end):
    with pytest.raises(ValueError):
        resolve_shard_range(15413, 16, start, end)


def test_slice_shard_range_keeps_short_final_shard():
    records = list(range(15413))
    selected, resolved = slice_shard_range(records, 16, 960, 964)
    assert resolved == ShardRange(960, 964, 964)
    assert selected[0] == 15360
    assert selected[-1] == 15412
```

- [ ] **Step 2: Run range tests and verify RED**

Run: `pytest -q tests/test_shard_ranges.py`

Expected: collection fails because `bimer.shard_ranges` does not exist.

- [ ] **Step 3: Implement range resolution**

```python
@dataclass(frozen=True, slots=True)
class ShardRange:
    start: int
    end: int
    total_shards: int

    @property
    def shard_count(self) -> int:
        return self.end - self.start


def resolve_shard_range(record_count, shard_size, start_shard, end_shard):
    if record_count < 0 or shard_size <= 0:
        raise ValueError("record_count must be non-negative and shard_size positive")
    total = (record_count + shard_size - 1) // shard_size
    if start_shard is None and end_shard is None:
        return ShardRange(0, total, total)
    if start_shard is None or end_shard is None:
        raise ValueError("start-shard and end-shard must be supplied together")
    if not 0 <= start_shard < end_shard <= total:
        raise ValueError(f"shard range must satisfy 0 <= start < end <= {total}")
    return ShardRange(start_shard, end_shard, total)
```

`slice_shard_range` resolves the range and returns the official-order slice.

- [ ] **Step 4: Write failing offset-numbering tests**

```python
def test_record_shards_applies_global_offset():
    shards = list(record_shards(make_records(33), 16, shard_index_offset=120))
    assert [index for index, _, _ in shards] == [120, 121, 122]
    assert [len(chunk) for _, chunk, _ in shards] == [16, 16, 1]


def test_parallel_config_rejects_negative_shard_offset():
    with pytest.raises(ValueError, match="shard_index_offset"):
        ParallelFeatureExtractionConfig(shard_index_offset=-1)
```

- [ ] **Step 5: Run offset tests and verify RED**

Run: `pytest -q tests/test_parallel_feature_extraction.py -k 'global_offset or negative_shard_offset'`

Expected: signature/configuration failures because offset support is absent.

- [ ] **Step 6: Propagate global offsets through the runner**

Add `shard_index_offset: int = 0` to the configuration, validate it as
non-negative, and pass it to every `record_shards` call in pending-stage,
completed-final, merge, and shard-count paths. Keep default zero behavior.

- [ ] **Step 7: Run Task 1 tests and commit**

Run: `pytest -q tests/test_shard_ranges.py tests/test_parallel_feature_extraction.py`

Expected: PASS.

```bash
git add src/bimer/shard_ranges.py src/bimer/parallel_feature_extraction.py tests/test_shard_ranges.py tests/test_parallel_feature_extraction.py
git commit -m "feat: add global shard range numbering"
```

---

### Task 2: Ranged Parallel Extraction CLI

**Files:**
- Modify: `src/bimer/cli.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: `slice_shard_range` and `ParallelFeatureExtractionConfig.shard_index_offset`.
- Produces: `extract-features --start-shard N --end-shard M`.

- [ ] **Step 1: Write failing parser and validation tests**

```python
def test_extract_parser_accepts_shard_range():
    args = build_parser().parse_args([
        "extract-features", "--manifest", "m.jsonl", "--features", "f",
        "--yunet-model", "y.onnx", "--dataset", "emotiontalk",
        "--split", "train", "--mode", "parallel",
        "--start-shard", "120", "--end-shard", "240",
    ])
    assert (args.start_shard, args.end_shard) == (120, 240)


def required_extract_args(tmp_path):
    return [
        "extract-features", "--manifest", str(tmp_path / "manifest.jsonl"),
        "--features", str(tmp_path / "features"),
        "--yunet-model", str(tmp_path / "yunet.onnx"),
        "--dataset", "emotiontalk", "--split", "train",
    ]


def test_range_requires_parallel_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "read_manifest", lambda _path: make_cli_records(16))
    with pytest.raises(ValueError, match="parallel"):
        cli.main(required_extract_args(tmp_path) + [
            "--start-shard", "0", "--end-shard", "1",
        ])


def test_range_requires_both_bounds(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "read_manifest", lambda _path: make_cli_records(16))
    monkeypatch.setattr(cli.torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match="supplied together"):
        cli.main(required_extract_args(tmp_path) + [
            "--mode", "parallel", "--start-shard", "0",
        ])
```

- [ ] **Step 2: Run CLI range tests and verify RED**

Run: `pytest -q tests/test_cli.py -k shard_range`

Expected: parser rejects unknown flags.

- [ ] **Step 3: Add parser flags and range preconditions**

Add optional integer flags. If either is supplied, require both, parallel mode,
dataset, and split before CUDA validation or model factories are built.

- [ ] **Step 4: Write failing slice/offset routing test**

```python
def test_parallel_range_routes_official_slice_and_offset(tmp_path, monkeypatch):
    records = make_cli_records(40)
    captured = install_fake_parallel_runner(monkeypatch, records)
    assert cli.main(range_args(tmp_path, start=1, end=3, shard_size=16)) == 0
    assert captured["records"] == records[16:40]
    assert captured["config"].shard_index_offset == 1
```

- [ ] **Step 5: Run routing test and verify RED**

Run: `pytest -q tests/test_cli.py::test_parallel_range_routes_official_slice_and_offset`

Expected: runner receives the full list or lacks the offset.

- [ ] **Step 6: Slice each explicit group before runner construction**

Resolve the official group before building its runner configuration. Preserve
existing behavior when range flags are absent. A range that selects no records
or exceeds the split fails before model construction.

- [ ] **Step 7: Run Task 2 tests and commit**

Run: `pytest -q tests/test_cli.py`

Expected: PASS.

```bash
git add src/bimer/cli.py tests/test_cli.py
git commit -m "feat: route global shard ranges from CLI"
```

---

### Task 3: Strict Feature-Range Verification and Completion Metadata

**Files:**
- Create: `src/bimer/feature_verification.py`
- Create: `tests/test_feature_verification.py`
- Modify: `src/bimer/cli.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Produces: `FeatureVerificationResult`, `verify_feature_range`, `write_range_completion`.
- Produces CLI: `bimer verify-features`.

- [ ] **Step 1: Write failing verifier success tests**

```python
def test_verify_complete_feature_range(tmp_path):
    records = make_records(33)
    write_valid_feature_shards(tmp_path, records, shard_size=16, offset=0)
    result = verify_feature_range(records, FeatureStore(tmp_path), shard_size=16)
    assert result.sample_count == 33
    assert result.verified_shards == 3
    assert result.is_valid is True


def test_verify_partial_range_allows_other_global_shards(tmp_path):
    records = make_records(40)
    write_valid_feature_shards(tmp_path, records[:16], shard_size=16, offset=0)
    write_valid_feature_shards(tmp_path, records[16:], shard_size=16, offset=1)
    selected, resolved = slice_shard_range(records, 16, 1, 3)
    result = verify_feature_range(
        selected, FeatureStore(tmp_path), shard_size=16,
        shard_index_offset=resolved.start, expected_shard_count=resolved.shard_count,
    )
    assert result.start_shard == 1
    assert result.end_shard == 3
```

- [ ] **Step 2: Run verifier tests and verify RED**

Run: `pytest -q tests/test_feature_verification.py`

Expected: collection fails because the verifier module does not exist.

- [ ] **Step 3: Implement result schema and successful verification**

For each manifest-derived global shard, require exactly one final file and call
`verified_final_shard` with expected IDs. Ensure concatenated IDs remain unique.
Return a dataclass with `to_dict()` for JSON output.

- [ ] **Step 4: Write failing corruption tests**

```python
@pytest.mark.parametrize("mutation,match", [
    (remove_middle_shard, "missing shard"),
    (reorder_ids, "unexpected sample IDs"),
    (duplicate_ids, "unique"),
    (wrong_text_width, "width 768"),
    (malformed_mask, "shape [rows, 3]"),
    (insert_nan, "finite"),
])
def test_verify_feature_range_rejects_invalid_cache(tmp_path, mutation, match):
    records = make_records(33)
    write_valid_feature_shards(tmp_path, records, 16, 0)
    mutation(tmp_path)
    with pytest.raises(ValueError, match=match):
        verify_feature_range(records, FeatureStore(tmp_path), shard_size=16)
```

- [ ] **Step 5: Implement strict failure checks and atomic completion JSON**

Write `features/ranges/range-00000-00120.json` through a `.tmp` file and
`Path.replace`. Only accept a successful result. Include dataset, split,
samples, expected/verified shards, start/end, total shards, and `is_valid`.

- [ ] **Step 6: Write failing CLI command test**

```python
def test_verify_features_command_prints_json_and_writes_completion(tmp_path, capsys):
    # Build manifest and valid range files.
    assert cli.main([
        "verify-features", "--manifest", str(manifest),
        "--features", str(tmp_path), "--dataset", "emotiontalk",
        "--split", "train", "--shard-size", "16",
        "--start-shard", "0", "--end-shard", "2",
        "--write-completion",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["is_valid"] is True
    assert (tmp_path / "ranges" / "range-00000-00002.json").is_file()
```

- [ ] **Step 7: Implement `verify-features` routing**

Read and filter one explicit group, resolve/slice its range, call the verifier,
optionally write completion metadata, and print UTF-8 JSON.

- [ ] **Step 8: Run Task 3 tests and commit**

Run: `pytest -q tests/test_feature_verification.py tests/test_cli.py`

Expected: PASS.

```bash
git add src/bimer/feature_verification.py src/bimer/cli.py tests/test_feature_verification.py tests/test_cli.py
git commit -m "feat: verify global feature ranges"
```

---

### Task 4: Kaggle Restore Workflow and Reproducibility Bundle

**Files:**
- Modify: `docs/kaggle.md`
- Modify: `README.md`
- Modify: `tests/test_kaggle_script.py`
- Rebuild: `data/processed/bimer-kaggle-source.zip`

**Interfaces:**
- Documents eight train commands and the saved-output restore snippet.

- [ ] **Step 1: Write failing documentation assertions**

```python
def test_kaggle_guide_documents_cross_session_train_ranges():
    guide = Path("docs/kaggle.md").read_text(encoding="utf-8")
    assert "--start-shard 0" in guide
    assert "--end-shard 120" in guide
    assert "--start-shard 840" in guide
    assert "--end-shard 964" in guide
    assert "verify-features" in guide
    assert "dirs_exist_ok=True" in guide
```

- [ ] **Step 2: Run documentation test and verify RED**

Run: `pytest -q tests/test_kaggle_script.py -k cross_session`

Expected: FAIL because the guide lacks the workflow.

- [ ] **Step 3: Document restore, extract, verify, and Quick Save**

Include the mounted path
`/kaggle/input/notebooks/zhoujunjie2/bimer-emotiontalk-bootstrap/features-emotiontalk-train-v4`,
the common working root, the eight ranges, verification commands, and the rule
that the next range starts only after completion JSON and Quick Save.

- [ ] **Step 4: Run documentation tests and rebuild safe bundle**

Run:

```bash
pytest -q tests/test_kaggle_script.py
git ls-files | zip -q -FS data/processed/bimer-kaggle-source.zip -@
unzip -t data/processed/bimer-kaggle-source.zip
```

Expected: PASS and no archive errors.

- [ ] **Step 5: Commit documentation**

```bash
git add docs/kaggle.md README.md tests/test_kaggle_script.py
git commit -m "docs: add cross-session train extraction workflow"
```

---

### Task 5: Full Verification and Kaggle First Range

**Files:**
- Verify all source, tests, docs, and bundle.
- Kaggle output: `/kaggle/working/features-emotiontalk-train-v4`.

**Interfaces:**
- Produces the first persisted range `[0, 120)` and evidence for later sessions.

- [ ] **Step 1: Run fresh local verification**

```bash
pytest -q
python -m compileall -q src/bimer
git diff --check
unzip -t data/processed/bimer-kaggle-source.zip
```

Expected: all tests pass and all commands exit zero.

- [ ] **Step 2: Deploy the safe bundle to Kaggle**

Install editable source in `/kaggle/working/bimer-v5`, run the complete test
suite, and confirm no token is present in the archive.

- [ ] **Step 3: Run a two-shard Kaggle range smoke**

Run `[0, 2)`, verify files are numbered 00000 and 00001, then repeat the same
command and confirm extractor factories are skipped.

- [ ] **Step 4: Run train range `[0, 120)`**

Use the common train feature root, shard size 16, dual-T4 settings, and the new
range arguments. After extraction, run `verify-features --write-completion`.

- [ ] **Step 5: Quick Save the completed range**

Name the Kaggle version `EmotionTalk train shards 0-119 complete` and confirm
the output contains `range-00000-00120.json`.

- [ ] **Step 6: Record the next range**

The following session restores the saved root and runs `[120, 240)`. Do not
start it until Version output is mounted and the restored `[0, 120)` range
passes verification.

## Final Verification Checklist

- [ ] Full extraction without range flags remains backward compatible.
- [ ] Invalid range requests fail before model construction.
- [ ] Global file numbers remain correct for every interval.
- [ ] Repeated ranges skip verified final shards.
- [ ] Partial verification allows completed shards outside its interval.
- [ ] Full verification requires exactly shards 0 through 963 and 15,413 IDs.
- [ ] Completion metadata is written only after successful verification.
- [ ] Kaggle restore keeps files and IDs unchanged.
- [ ] No secret or raw media enters the source bundle.
