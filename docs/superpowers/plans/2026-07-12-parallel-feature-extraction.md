# Parallel Multimodal Feature Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable dual-GPU feature pipeline that overlaps CPU decoding with batched inference and preserves the existing final `FeatureShard` contract.

**Architecture:** Text and audio run sequentially on GPU 0 while vision runs concurrently on GPU 1. Each modality writes deterministic atomic staging shards; a strict merger validates and joins them by manifest-ordered `sample_id`, while verified legacy final shards are reused.

**Tech Stack:** Python 3.11, NumPy 2.x, PyTorch 2.x, Transformers 4.x, torchvision R3D-18, OpenCV YuNet, FFmpeg, `concurrent.futures`, pytest 8.x.

## Global Constraints

- Final feature dimensions stay text 768, audio 1024, vision 512.
- Preserve official split and manifest order; never randomly repartition utterances.
- GPU defaults: text/audio `cuda:0`, vision `cuda:1`.
- Batch defaults: text 64, audio 8, vision 8; CPU workers: audio 4, vision 4.
- Fewer than 4 detected face frames means zero vision features and `vision_mask=False`.
- Empty audio means zero audio features and `audio_mask=False`; decode failures remain errors.
- CUDA OOM halves the current batch until 1, then raises.
- Existing final shards are reused only after IDs, dimensions, masks and finite values pass validation.
- Old/new fixed-sample tolerance is `rtol=1e-4`, `atol=1e-5`.
- Keep serial mode and existing outputs as the rollback path.
- Never print or persist `HF_TOKEN`.

## File Map

- Create `src/bimer/modality_store.py`: modality staging schema, atomic storage and strict merge.
- Create `src/bimer/parallel_feature_extraction.py`: bounded CPU prefetch, adaptive batching and orchestration.
- Modify `src/bimer/feature_store.py`: generic shape/finite validation and atomic final writes.
- Modify `src/bimer/feature_extractors.py`: prepared clips and batched R3D inference.
- Modify `src/bimer/cli.py`: serial/parallel routing and resource flags.
- Create `tests/test_modality_store.py` and `tests/test_parallel_feature_extraction.py`.
- Modify `tests/test_feature_store.py`, `tests/test_feature_extractors.py`, `tests/test_cli.py`.
- Modify `README.md`, `docs/kaggle.md`, `scripts/prepare_emotiontalk_kaggle.sh`, `tests/test_kaggle_script.py`.

---

### Task 1: Validated Atomic Final and Staging Stores

**Files:**
- Modify: `src/bimer/feature_store.py`
- Create: `src/bimer/modality_store.py`
- Modify: `tests/test_feature_store.py`
- Create: `tests/test_modality_store.py`

**Interfaces:**
- Consumes: existing `FeatureShard`, `FeatureStore`, dataset, split and manifest IDs.
- Produces: `validate_feature_shard`, `ModalityShard`, `ModalityStore`, `verified_final_shard`, `merge_staged_shard`.

- [ ] **Step 1: Write failing final-store tests**

Add these cases to `tests/test_feature_store.py`:

```python
def test_feature_shard_rejects_non_matrix_features():
    with pytest.raises(ValueError, match="text must be a matrix"):
        FeatureShard(
            sample_ids=np.array(["one"]), text=np.ones(768, np.float32),
            audio=np.ones((1, 6), np.float32),
            vision=np.ones((1, 5), np.float32),
            modality_mask=np.ones((1, 3), np.bool_),
        )


def test_feature_shard_rejects_duplicate_ids_and_non_finite_values():
    values = np.ones((2, 768), np.float32)
    values[0, 0] = np.nan
    with pytest.raises(ValueError, match="unique|finite"):
        FeatureShard(
            sample_ids=np.array(["same", "same"]), text=values,
            audio=np.ones((2, 1024), np.float32),
            vision=np.ones((2, 512), np.float32),
            modality_mask=np.ones((2, 3), np.bool_),
        )


def test_feature_store_write_leaves_no_temporary_file(tmp_path):
    store = FeatureStore(tmp_path)
    store.write("emotiontalk", "validation", 0, make_feature_shard(["one"]))
    assert store.path("emotiontalk", "validation", 0).is_file()
    assert not list(tmp_path.rglob("*.tmp"))
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_feature_store.py -v`

Expected: FAIL because matrix shape, unique IDs, finite values and atomic writing are not enforced.

- [ ] **Step 3: Implement final-shard validation and atomic write**

Add to `src/bimer/feature_store.py`:

```python
def validate_feature_shard(
    shard: FeatureShard,
    expected_dims: Mapping[str, int] | None = None,
) -> None:
    ids = shard.sample_ids.astype(str)
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("sample_ids must be unique")
    for name in ("text", "audio", "vision"):
        values = np.asarray(getattr(shard, name))
        if values.ndim != 2:
            raise ValueError(f"{name} must be a matrix")
        if values.shape[0] != len(ids):
            raise ValueError(f"{name} must have one row per sample")
        if expected_dims is not None and values.shape[1] != expected_dims[name]:
            raise ValueError(f"{name} must have width {expected_dims[name]}")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} features must be finite")
```

Call it without `expected_dims` from `FeatureShard.__post_init__`, preserving small synthetic dimensions used by model tests. The extraction merger calls it with `MODALITY_DIMS` so real cached features must be 768/1024/512. In `FeatureStore.write`, write to `path.with_suffix(".npz.tmp")` through `temporary.open("wb")`, then call `temporary.replace(path)` in a `try/finally` that removes a leftover temporary file.

- [ ] **Step 4: Write failing staging and merge tests**

Create `tests/test_modality_store.py`:

```python
def test_modality_store_round_trips_and_validates_expected_ids(tmp_path):
    store = ModalityStore(tmp_path, "text", 768)
    shard = ModalityShard(
        np.array(["a", "b"]), np.ones((2, 768), np.float32),
        np.array([True, True]),
    )
    path = store.write("emotiontalk", "validation", 0, shard)
    loaded = store.read_verified(
        "emotiontalk", "validation", 0, np.array(["a", "b"])
    )
    assert path.is_file()
    assert loaded.sample_ids.tolist() == ["a", "b"]
    assert not list(tmp_path.rglob("*.tmp"))


def test_merge_staged_shard_rejects_reordered_ids(tmp_path):
    write_staging_triplet(tmp_path, text_ids=["a", "b"], audio_ids=["b", "a"], vision_ids=["a", "b"])
    with pytest.raises(ValueError, match="unexpected sample IDs"):
        merge_staged_shard(
            staging_root=tmp_path, final_store=FeatureStore(tmp_path / "final"),
            dataset="emotiontalk", split="validation", shard_index=0,
            expected_sample_ids=np.array(["a", "b"]),
        )


def test_merge_staged_shard_zeroes_unavailable_rows(tmp_path):
    write_staging_triplet(tmp_path, vision_available=[True, False])
    path = merge_staged_shard(
        staging_root=tmp_path, final_store=FeatureStore(tmp_path / "final"),
        dataset="emotiontalk", split="validation", shard_index=0,
        expected_sample_ids=np.array(["a", "b"]),
    )
    shard = FeatureStore(tmp_path / "final").read(path)
    assert shard.modality_mask.tolist() == [[True, True, True], [True, True, False]]
    assert np.all(shard.vision[1] == 0)
```

- [ ] **Step 5: Verify staging RED**

Run: `pytest tests/test_modality_store.py -v`

Expected: collection ERROR because `bimer.modality_store` does not exist.

- [ ] **Step 6: Implement staging schema, store and merge**

Create `src/bimer/modality_store.py` with these public definitions:

```python
MODALITY_DIMS = {"text": 768, "audio": 1024, "vision": 512}


@dataclass(frozen=True, slots=True)
class ModalityShard:
    sample_ids: np.ndarray
    features: np.ndarray
    available: np.ndarray


class ModalityStore:
    def __init__(self, root: Path | str, modality: str, output_dim: int) -> None:
        self.root, self.modality, self.output_dim = Path(root), modality, output_dim

    def path(self, dataset: str, split: str, shard_index: int) -> Path:
        return self.root / "staging" / dataset / split / self.modality / f"features-{shard_index:05d}.npz"

    def validate(self, shard: ModalityShard, expected_ids: np.ndarray | None = None) -> None:
        ids = shard.sample_ids.astype(str)
        if len(set(ids.tolist())) != len(ids):
            raise ValueError("sample_ids must be unique")
        if expected_ids is not None and not np.array_equal(ids, expected_ids.astype(str)):
            raise ValueError("staging shard has unexpected sample IDs")
        if shard.features.shape != (len(ids), self.output_dim):
            raise ValueError(f"features must have shape [rows, {self.output_dim}]")
        if shard.available.shape != (len(ids),):
            raise ValueError("available must have shape [rows]")
        if not np.isfinite(shard.features).all():
            raise ValueError("features must be finite")
```

Implement `write`, `read`, and `read_verified` with `allow_pickle=False` and the same atomic pattern as the final store. `verified_final_shard` reads and validates an existing final path and rejects unexpected IDs. `merge_staged_shard` loads all three expected staging shards, zeroes rows where `available=False`, stacks masks in text/audio/vision order, and writes one `FeatureShard`.

- [ ] **Step 7: Verify GREEN and commit**

Run: `pytest tests/test_feature_store.py tests/test_modality_store.py -v`

Expected: PASS.

```bash
git add src/bimer/feature_store.py src/bimer/modality_store.py tests/test_feature_store.py tests/test_modality_store.py
git commit -m "feat: add validated atomic feature staging"
```

---

### Task 2: Batched R3D Visual Extraction

**Files:**
- Modify: `src/bimer/feature_extractors.py`
- Modify: `tests/test_feature_extractors.py`

**Interfaces:**
- Consumes: `read_uniform_video_frames`, `YuNetFaceCropper`, R3D-18.
- Produces: `prepare_video_clip`, `_prepare_clip_tensor`, `VisionFeatureExtractor.encode_clips`; keeps `encode_frames` and `encode_video` compatible.

- [ ] **Step 1: Write failing batch and availability tests**

```python
def test_vision_extractor_batches_clips_in_order():
    class RecordingVision(VisionFeatureExtractor):
        def __init__(self):
            self.sizes = []
        def _encode_clip_batch(self, clips):
            self.sizes.append(len(clips))
            return np.asarray([[clip[0, 0, 0, 0]] * 512 for clip in clips], np.float32)
    clips = [np.full((16, 112, 112, 3), value, np.uint8) for value in range(5)]
    extractor = RecordingVision()
    result = extractor.encode_clips(clips, batch_size=2)
    assert extractor.sizes == [2, 2, 1]
    assert result[:, 0].tolist() == [0, 1, 2, 3, 4]


def test_prepare_video_clip_marks_three_faces_unavailable(monkeypatch):
    frames = np.ones((16, 8, 8, 3), np.uint8)
    monkeypatch.setattr("bimer.feature_extractors.read_uniform_video_frames", lambda *_a, **_k: frames)
    clip, available = prepare_video_clip("video.mp4", face_cropper=FakeCropper(three_faces=True))
    assert clip.shape == (16, 112, 112, 3)
    assert available is False
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_feature_extractors.py -v`

Expected: FAIL because the new functions do not exist.

- [ ] **Step 3: Implement preparation and batch inference**

Use these signatures and tensor layout:

```python
def prepare_video_clip(
    video_path: Path | str, *, face_cropper: YuNetFaceCropper,
    frame_drop_fraction: float = 0.0, seed: int = 42,
) -> tuple[np.ndarray, bool]:
    frames = read_uniform_video_frames(video_path, count=16)
    if frame_drop_fraction:
        frames = drop_video_frames(frames, fraction=frame_drop_fraction, seed=seed)
    prepared, detected = [], []
    for frame in frames:
        crop, found = face_cropper.crop_largest(frame)
        prepared.append(np.asarray(Image.fromarray(crop).resize((112, 112))))
        detected.append(found)
    return np.stack(prepared), vision_modality_available(detected)


def _prepare_clip_tensor(clips: Sequence[np.ndarray]) -> Tensor:
    tensor = torch.from_numpy(np.stack(clips).copy()).permute(0, 1, 4, 2, 3).float() / 255.0
    batch, frames, channels, height, width = tensor.shape
    tensor = torch.nn.functional.interpolate(
        tensor.reshape(batch * frames, channels, height, width),
        size=(112, 112), mode="bilinear", align_corners=False,
    ).reshape(batch, frames, channels, 112, 112)
    mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 1, 3, 1, 1)
    return ((tensor - mean) / std).permute(0, 2, 1, 3, 4)
```

`_encode_clip_batch` moves the tensor to `self.device` and runs the model. `encode_clips` slices by `batch_size` and concatenates. `encode_frames` delegates to one clip; `encode_video` calls `prepare_video_clip`, returns zeros if unavailable, otherwise encodes one clip.

- [ ] **Step 4: Verify GREEN and commit**

Run: `pytest tests/test_feature_extractors.py tests/test_inference.py -v`

Expected: PASS.

```bash
git add src/bimer/feature_extractors.py tests/test_feature_extractors.py
git commit -m "feat: batch R3D visual extraction"
```

---

### Task 3: Adaptive GPU Batching and Bounded CPU Prefetch

**Files:**
- Create: `src/bimer/parallel_feature_extraction.py`
- Create: `tests/test_parallel_feature_extraction.py`

**Interfaces:**
- Produces: `encode_adaptive`, `prefetched_map`, `CpuWorkerError`, `record_shards`.

- [ ] **Step 1: Write failing helper tests**

```python
def test_encode_adaptive_halves_cuda_oom_batch():
    encoder = OomAboveTwoEncoder()
    result = encode_adaptive(encoder, [1, 2, 3], initial_batch_size=8)
    assert encoder.attempts == [8, 4, 2]
    assert result.shape == (3, 3)


def test_encode_adaptive_reraises_non_oom():
    with pytest.raises(RuntimeError, match="bad model"):
        encode_adaptive(BrokenEncoder(), [1], initial_batch_size=8)


def test_prefetched_map_preserves_order_and_propagates_failure():
    assert list(prefetched_map(
        square, range(5), workers=2, queue_capacity=2,
        executor_factory=ThreadPoolExecutor,
    )) == [0, 1, 4, 9, 16]
    with pytest.raises(CpuWorkerError, match="input 2"):
        list(prefetched_map(
            fail_on_two, range(4), workers=2, queue_capacity=2,
            executor_factory=ThreadPoolExecutor,
        ))
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_parallel_feature_extraction.py -v`

Expected: collection ERROR because the module does not exist.

- [ ] **Step 3: Implement adaptive batching**

```python
def encode_adaptive(encoder, values, *, initial_batch_size: int) -> np.ndarray:
    batch_size = initial_batch_size
    while True:
        try:
            return encoder.encode(values, batch_size=batch_size)
        except BaseException as error:
            is_oom = isinstance(error, torch.cuda.OutOfMemoryError) or (
                isinstance(error, RuntimeError) and "out of memory" in str(error).lower()
            )
            if not is_oom or batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
```

- [ ] **Step 4: Implement ordered bounded prefetch**

`prefetched_map` must maintain at most `workers + queue_capacity` futures. It submits initial work, yields `future.result()` in input order, submits one replacement per yield, cancels pending futures on exit, and wraps worker failures as:

```python
class CpuWorkerError(RuntimeError):
    def __init__(self, index: int, error: BaseException) -> None:
        super().__init__(f"CPU worker failed at input {index}: {error}")
        self.index = index


def record_shards(records: Sequence[UtteranceRecord], shard_size: int):
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    for index, start in enumerate(range(0, len(records), shard_size)):
        chunk = records[start:start + shard_size]
        ids = np.asarray([record.sample_id for record in chunk], dtype=str)
        yield index, chunk, ids
```

Default `executor_factory` is `ProcessPoolExecutor`; tests inject `ThreadPoolExecutor` to avoid multiprocessing fixture pickling.

- [ ] **Step 5: Verify GREEN and commit**

Run: `pytest tests/test_parallel_feature_extraction.py -v`

Expected: PASS.

```bash
git add src/bimer/parallel_feature_extraction.py tests/test_parallel_feature_extraction.py
git commit -m "feat: add adaptive batches and bounded prefetch"
```

---

### Task 4: Resumable Modality Stages and Dual-Branch Runner

**Files:**
- Modify: `src/bimer/parallel_feature_extraction.py`
- Modify: `tests/test_parallel_feature_extraction.py`

**Interfaces:**
- Produces: `ParallelFeatureExtractionConfig`, `extract_text_stage`, `extract_audio_stage`, `extract_vision_stage`, `ParallelFeatureExtractionRunner.run`.

- [ ] **Step 1: Write failing resume, mask and ordering tests**

```python
def test_text_stage_skips_verified_staging_shard(tmp_path):
    records = make_records(2)
    write_existing_text_staging(tmp_path, records)
    extract_text_stage(records, tmp_path, MustNotConstruct, shard_size=2, batch_size=64)


def test_audio_stage_preserves_order_and_masks_empty_audio(tmp_path):
    records = make_records(3)
    extract_audio_stage(
        records, tmp_path, FakeAudioExtractor,
        waveform_loader=WaveformsWithEmptySecond(records),
        shard_size=3, batch_size=2, workers=2, queue_capacity=2,
        executor_factory=ThreadPoolExecutor,
    )
    shard = ModalityStore(tmp_path, "audio", 1024).read_verified(
        "emotiontalk", "validation", 0,
        np.array([record.sample_id for record in records]),
    )
    assert shard.available.tolist() == [True, False, True]
    assert np.all(shard.features[1] == 0)


def test_vision_stage_encodes_only_available_clips(tmp_path):
    records = make_records(3)
    extract_vision_stage(
        records, tmp_path, RecordingVisionExtractor,
        prepared_loader=PreparedClips([True, False, True]),
        shard_size=3, batch_size=8, workers=2, queue_capacity=2,
        executor_factory=ThreadPoolExecutor,
    )
    shard = read_vision_staging(tmp_path, records)
    assert shard.available.tolist() == [True, False, True]
    assert np.all(shard.features[1] == 0)
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_parallel_feature_extraction.py -v`

Expected: FAIL because stage functions do not exist.

- [ ] **Step 3: Implement the three stages**

Each stage iterates `record_shards`, calls `ModalityStore.read_verified` when its target exists, and constructs the extractor only when work remains. Text encodes record text. Audio consumes ordered waveforms from `prefetched_map`, uses a 160-sample zero input for empty audio, then zeroes unavailable output rows. Vision consumes `(clip, available)`, encodes only available clips, restores them to an all-zero `[rows, 512]` matrix, and writes availability.

Enforce this check before every write:

```python
def validate_stage_output(modality: str, features: np.ndarray, rows: int, width: int) -> np.ndarray:
    output = np.asarray(features, np.float32)
    if output.shape != (rows, width):
        raise ValueError(f"{modality} returned {output.shape}, expected {(rows, width)}")
    if not np.isfinite(output).all():
        raise ValueError(f"{modality} returned non-finite features")
    return output
```

When CPU preprocessing raises, append one UTF-8 JSON object with `modality`, `sample_id`, `path`, `error_type`, and `message` to `staging/errors.jsonl`, then re-raise without writing that shard.

- [ ] **Step 4: Write failing concurrency and merge-gate tests**

```python
def test_runner_overlaps_gpu_branches_and_merges_after_both(tmp_path):
    events = SynchronizedEvents()
    runner = make_injected_runner(events)
    runner.run(make_records(2), FeatureStore(tmp_path / "final"))
    assert events.overlapped("text_audio", "vision")
    assert events.before("audio_done", "merge")
    assert events.before("vision_done", "merge")


def test_runner_does_not_merge_after_branch_failure(tmp_path):
    runner = make_failing_runner(RuntimeError("audio failed"))
    with pytest.raises(RuntimeError, match="audio failed"):
        runner.run(make_records(2), FeatureStore(tmp_path / "final"))
    assert runner.merge_calls == 0
```

- [ ] **Step 5: Verify concurrency RED**

Run: `pytest tests/test_parallel_feature_extraction.py -v`

Expected: FAIL because the runner does not exist.

- [ ] **Step 6: Implement configuration and orchestration**

```python
@dataclass(frozen=True, slots=True)
class ParallelFeatureExtractionConfig:
    shard_size: int = 1024
    text_batch_size: int = 64
    audio_batch_size: int = 8
    vision_batch_size: int = 8
    audio_workers: int = 4
    vision_workers: int = 4
    queue_capacity: int = 8


class ParallelFeatureExtractionRunner:
    def run(self, records, final_store):
        validate_one_dataset_split(records)
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="bimer-gpu") as pool:
            gpu0 = pool.submit(self._run_text_then_audio, records)
            gpu1 = pool.submit(self._run_vision, records)
            gpu0.result()
            gpu1.result()
        return self._merge_all(records, final_store)
```

`_run_text_then_audio` calls the text stage, releases its extractor and CUDA cache, then calls audio. `_run_vision` uses its independent factory/device. `_merge_all` iterates every original manifest shard and calls `merge_staged_shard`, which returns verified legacy final files without requiring staging.

- [ ] **Step 7: Add deterministic old/new integration test**

```python
def test_parallel_matches_serial_for_sixteen_samples(tmp_path):
    records = make_records(16)
    old = run_serial_deterministic(records, tmp_path / "old")
    new = run_parallel_deterministic(records, tmp_path / "new")
    assert new.sample_ids.tolist() == old.sample_ids.tolist()
    np.testing.assert_array_equal(new.modality_mask, old.modality_mask)
    np.testing.assert_allclose(new.text, old.text, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(new.audio, old.audio, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(new.vision, old.vision, rtol=1e-4, atol=1e-5)
```

- [ ] **Step 8: Verify GREEN and commit**

Run: `pytest tests/test_parallel_feature_extraction.py tests/test_modality_store.py tests/test_feature_store.py -v`

Expected: PASS.

```bash
git add src/bimer/parallel_feature_extraction.py tests/test_parallel_feature_extraction.py
git commit -m "feat: add resumable dual GPU extraction runner"
```

---

### Task 5: CLI, Kaggle Documentation and Packaging

**Files:**
- Modify: `src/bimer/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `docs/kaggle.md`
- Modify: `README.md`
- Modify: `scripts/prepare_emotiontalk_kaggle.sh`
- Modify: `tests/test_kaggle_script.py`

**Interfaces:**
- Produces: `bimer extract-features --mode parallel` with dual-T4 defaults; serial remains default.

- [ ] **Step 1: Write failing CLI tests**

```python
def test_parallel_feature_defaults_target_dual_t4():
    args = build_parser().parse_args([
        "extract-features", "--manifest", "m.jsonl", "--features", "features",
        "--yunet-model", "yunet.onnx", "--mode", "parallel",
    ])
    assert args.text_audio_device == "cuda:0"
    assert args.vision_device == "cuda:1"
    assert (args.text_batch_size, args.audio_batch_size, args.vision_batch_size) == (64, 8, 8)
    assert (args.audio_workers, args.vision_workers, args.queue_capacity) == (4, 4, 8)


def test_serial_feature_mode_remains_default():
    args = build_parser().parse_args([
        "extract-features", "--manifest", "m.jsonl", "--features", "features",
        "--yunet-model", "yunet.onnx",
    ])
    assert args.mode == "serial"
```

- [ ] **Step 2: Verify CLI RED**

Run: `pytest tests/test_cli.py -v`

Expected: FAIL because the arguments do not exist.

- [ ] **Step 3: Add parser flags and route modes**

Add:

```python
extract.add_argument("--mode", choices=["serial", "parallel"], default="serial")
extract.add_argument("--text-audio-device", default="cuda:0")
extract.add_argument("--vision-device", default="cuda:1")
extract.add_argument("--text-batch-size", type=int, default=64)
extract.add_argument("--audio-batch-size", type=int, default=8)
extract.add_argument("--vision-batch-size", type=int, default=8)
extract.add_argument("--audio-workers", type=int, default=4)
extract.add_argument("--vision-workers", type=int, default=4)
extract.add_argument("--queue-capacity", type=int, default=8)
extract.add_argument("--staging")
```

Keep the current serial body unchanged. Parallel mode validates the two requested CUDA devices, uses `args.staging or args.features`, builds lazy text/audio/vision factories for their assigned devices, builds top-level picklable audio and vision CPU worker callables, and runs `ParallelFeatureExtractionRunner` for each dataset/split group.

- [ ] **Step 4: Add failing packaging/documentation test**

```python
def test_kaggle_resources_include_parallel_pipeline():
    script = Path("scripts/prepare_emotiontalk_kaggle.sh").read_text()
    guide = Path("docs/kaggle.md").read_text()
    assert "parallel_feature_extraction.py" in script
    assert "modality_store.py" in script
    assert "--mode parallel" in guide
    assert "--text-audio-device cuda:0" in guide
    assert "--vision-device cuda:1" in guide
```

- [ ] **Step 5: Verify packaging RED**

Run: `pytest tests/test_kaggle_script.py -v`

Expected: FAIL because the new pipeline is not yet documented or packaged.

- [ ] **Step 6: Document the exact Kaggle command and rollback**

Add this command to `docs/kaggle.md` and summarize it in `README.md`:

```bash
bimer extract-features \
  --manifest /kaggle/working/bimer-output/emotiontalk.jsonl \
  --features /kaggle/working/features-emotiontalk-validation-v2 \
  --yunet-model /kaggle/working/yunet.onnx \
  --dataset emotiontalk --split validation --mode parallel \
  --text-audio-device cuda:0 --vision-device cuda:1 \
  --text-batch-size 64 --audio-batch-size 8 --vision-batch-size 8 \
  --audio-workers 4 --vision-workers 4 --queue-capacity 8 \
  --shard-size 16
```

Document `watch -n 2 nvidia-smi`, staging paths, identical-command resume, and rollback with `--mode serial --device cuda`. Update the existing archive list so both new source files and tests enter `data/processed/bimer-kaggle-source.zip`.

- [ ] **Step 7: Verify CLI and resources, then commit**

Run: `pytest tests/test_cli.py tests/test_kaggle_script.py -v`

Expected: PASS.

```bash
git add src/bimer/cli.py tests/test_cli.py README.md docs/kaggle.md scripts/prepare_emotiontalk_kaggle.sh tests/test_kaggle_script.py
git commit -m "docs: add dual GPU Kaggle extraction workflow"
```

---

### Task 6: Full Local and Kaggle Verification

**Files:**
- Verify: all changed source, tests, docs and `data/processed/bimer-kaggle-source.zip`.

**Interfaces:**
- Produces: a locally verified bundle and measured 32-sample/full-validation Kaggle evidence.

- [ ] **Step 1: Run complete local verification**

```bash
pytest -q
git diff --check
python -m compileall -q src/bimer
```

Expected: all tests PASS, no whitespace errors, compilation succeeds.

- [ ] **Step 2: Rebuild and inspect the Kaggle source bundle**

Run the repository's safe bundle build command from `scripts/prepare_emotiontalk_kaggle.sh`, then inspect without exposing secrets:

```bash
unzip -l data/processed/bimer-kaggle-source.zip | rg 'modality_store|parallel_feature_extraction'
```

Expected: both source modules and both test modules are listed.

- [ ] **Step 3: Run the 32-sample Kaggle smoke test**

Create a manifest containing the first 32 validation records without changing their order. Run the documented parallel command against that manifest. Record both GPU utilization, peak memory, each modality's throughput, total wall time, ID order, dimensions, masks and finite-value checks.

Expected: GPU 0 performs text/audio, GPU 1 performs vision, all 32 samples merge, no NaN/Inf is present.

- [ ] **Step 4: Verify Kaggle interruption and resume**

Interrupt after at least one staging shard is complete. Run the identical command again. Verify logs skip the completed shard and its checksum remains unchanged.

Expected: the run resumes from missing modality shards without recomputing verified files.

- [ ] **Step 5: Compare fixed real samples with the serial output**

Load matching old/new samples and execute:

```python
np.testing.assert_array_equal(new.sample_ids, old.sample_ids)
np.testing.assert_array_equal(new.modality_mask, old.modality_mask)
np.testing.assert_allclose(new.text, old.text, rtol=1e-4, atol=1e-5)
np.testing.assert_allclose(new.audio, old.audio, rtol=1e-4, atol=1e-5)
np.testing.assert_allclose(new.vision, old.vision, rtol=1e-4, atol=1e-5)
```

Expected: all comparisons PASS.

- [ ] **Step 6: Run full validation only after smoke/resume/comparison pass**

Run the full validation command, report measured wall time and stage bottlenecks, and preserve the old serial output until final validation succeeds. Do not claim the 60-minute target unless the measurement supports it.

- [ ] **Step 7: Final commit for rebuilt reproducibility artifact**

```bash
git add data/processed/bimer-kaggle-source.zip
git commit -m "build: refresh Kaggle feature extraction bundle"
```

## Final Verification Checklist

- [ ] Serial mode remains available.
- [ ] Verified final shards skip all model construction.
- [ ] Each modality resumes independently.
- [ ] Wrong IDs, dimensions, NaN, Inf and corrupt staging files stop merging.
- [ ] Missing one or two modalities yields finite zero rows and correct masks.
- [ ] CPU worker failure creates an error record and no final shard.
- [ ] Fixed sample values match within `rtol=1e-4`, `atol=1e-5`.
- [ ] Both T4 GPUs show work on their assigned branches.
- [ ] Full validation duration and per-stage throughput are measured.
- [ ] Existing old outputs remain untouched until optimized validation passes.
