# Cross-Session Global Shard Extraction Design

## Goal

Enable EmotionTalk's 15,413-record training split to be extracted across
multiple Kaggle sessions without renumbering, overwriting, or manually merging
feature shards.

## Selected Approach

Add global shard-range support to the existing parallel feature extractor.
Each run processes an exclusive shard interval `[start, end)` while retaining
the shard numbers from the complete official split. All sessions therefore
write compatible files into the same feature root.

Separate per-session output roots were rejected because they require a later
renaming and merge step. Saving a single uninterrupted run was rejected because
Kaggle session duration and manual Quick Save timing are not reliable enough
for the full training split.

## Command-Line Contract

`bimer extract-features` gains two optional arguments:

```text
--start-shard INTEGER
--end-shard INTEGER
```

Both arguments must either be omitted or supplied together. `end-shard` is
exclusive. Ranged extraction is supported only with `--mode parallel` and
requires explicit `--dataset` and `--split` values.

For a selected official split with `N` records and shard size `S`, the total
number of shards is `ceil(N / S)`. A valid interval satisfies:

```text
0 <= start_shard < end_shard <= ceil(N / S)
```

The command slices records using:

```python
records[start_shard * shard_size : min(end_shard * shard_size, len(records))]
```

and passes `start_shard` as the global shard offset. Without the two arguments,
existing full-split behavior is unchanged.

## Extraction Architecture

`ParallelFeatureExtractionConfig` gains `shard_index_offset: int = 0`.
`record_shards` accepts the same offset and yields:

```python
(shard_index_offset + local_index, records, sample_ids)
```

Every staging, resume, merge, and final-store operation uses these global
indices. For example, the interval `[120, 240)` writes
`features-00120.npz` through `features-00239.npz`; it never writes shard zero.

The model scheduling remains unchanged: text and audio models initialize on
the main thread, then audio inference overlaps visual processing on the two T4
GPUs. Existing atomic writes and strict ID validation remain the publication
boundary.

## Cross-Session Data Flow

With `shard-size=16`, EmotionTalk train contains 964 shards numbered 0 through
963. The operational ranges are:

```text
[0, 120) [120, 240) [240, 360) [360, 480)
[480, 600) [600, 720) [720, 840) [840, 964)
```

Each interval processes at most 1,920 records and is expected to fit within a
single Kaggle session based on the measured test-split duration.

After an interval finishes:

1. Verify the interval's final files and record IDs.
2. Write `ranges/range-START-END.json` only after verification succeeds.
3. Use Kaggle Quick Save to persist `/kaggle/working`.
4. In the next session, copy the saved train feature root from the notebook's
   mounted output into `/kaggle/working` with `shutil.copytree(...,
   dirs_exist_ok=True)`.
5. Run the next interval against the same working feature root.

Repeating a completed interval is safe: verified final shards are skipped.
Partially completed text, audio, or visual staging shards are resumed
independently.

## Verification Contract

Add a `verify-features` command with these required inputs:

```text
--manifest PATH
--features PATH
--dataset DATASET
--split SPLIT
--shard-size INTEGER
```

Optional `--start-shard` and `--end-shard` verify one interval using the same
range rules. Without a range, the command verifies the complete selected split.

Verification must reject:

- missing or extra shard indices;
- sample IDs that differ from official manifest order;
- duplicate sample IDs;
- text, audio, or vision dimensions other than 768, 1024, and 512;
- malformed modality masks;
- NaN or infinite feature values.

Successful output is JSON containing dataset, split, sample count, expected and
verified shard counts, start/end shard, and `is_valid: true`.

## Failure Handling

Invalid ranges fail before constructing models or creating output files.
Existing shards with wrong IDs or corrupt arrays stop extraction rather than
being overwritten. A range-completion JSON file is never written when
extraction or verification fails. Existing atomic `.tmp` replacement behavior
continues to protect completed shards from interrupted writes.

## Testing

Automated tests cover:

1. Offset shard numbering, including the shorter final shard.
2. CLI rejection of incomplete, negative, reversed, out-of-bounds, serial-mode,
   and non-explicit dataset/split ranges.
3. CLI slicing and propagation of the global offset into the runner.
4. Two disjoint ranges writing into one feature root without collisions.
5. Re-running a completed range without constructing feature extractors.
6. Full and ranged verification, including missing shard, wrong ID, wrong
   dimension, malformed mask, duplicate ID, and non-finite failures.
7. Backward compatibility for full parallel and serial extraction commands.

## Acceptance Criteria

- Eight documented intervals cover every integer shard index from 0 through
  963 exactly once.
- Completed intervals survive Quick Save and can be restored into a later
  session without renaming files.
- Final train verification reports 15,413 samples and 964 shards.
- Existing validation and test feature roots remain unchanged.
- No Hugging Face token or raw dataset file is persisted in the repository or
  source bundle.
