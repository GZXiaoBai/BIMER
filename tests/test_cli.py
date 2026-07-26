import argparse
import json

import numpy as np
import pytest

import bimer.cli as cli
from bimer.cli import build_parser
from bimer.feature_store import FeatureShard, FeatureStore
from bimer.integrity import write_sha256_manifest
from bimer.labels import EMOTION_LABELS
from bimer.runtime import runtime_devices
from bimer.schema import UtteranceRecord


def make_cli_records(count, *, split="train"):
    return [
        UtteranceRecord(
            dataset="emotiontalk",
            split=split,
            dialogue_id="d1",
            utterance_id=index,
            text=f"line {index}",
            emotion="neutral",
            language="zh",
            start_seconds=float(index),
            end_seconds=float(index + 1),
            video_path=f"{index}.mp4",
        )
        for index in range(count)
    ]


def test_cli_exposes_all_reproducible_workflow_commands():
    parser = build_parser()
    expected = {
        "prepare-meld",
        "prepare-emotiontalk",
        "prepare-emotiontalk-official",
        "asr-manifest",
        "validate",
        "extract-features",
        "verify-features",
        "feature-stats",
        "overfit-smoke",
        "sample-corruption-manifest",
        "attach-quality",
        "train",
        "evaluate",
        "analyze",
        "serve",
        "doctor",
        "verify-evidence",
    }
    subparsers = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    assert set(subparsers.choices) == expected


def test_train_cli_exposes_v3_classification_loss_options():
    args = build_parser().parse_args(
        [
            "train",
            "--manifest",
            "manifest.jsonl",
            "--features",
            "features",
            "--output",
            "results",
            "--classification-loss",
            "focal",
            "--focal-gamma",
            "1.5",
            "--augmentation-modality",
            "audio",
            "--augmentation-severity",
            "10",
            "--gate-ranking-weight",
            "0.1",
            "--skip-test",
        ]
    )

    assert args.classification_loss == "focal"
    assert args.focal_gamma == 1.5
    assert args.augmentation_modality == ["audio"]
    assert args.augmentation_severity == [10.0]
    assert args.gate_ranking_weight == 0.1


def test_feature_stats_command_writes_report(tmp_path, monkeypatch, capsys):
    records = make_cli_records(2)
    store = FeatureStore(tmp_path / "features")
    store.write(
        "emotiontalk",
        "train",
        0,
        FeatureShard(
            sample_ids=np.asarray([record.sample_id for record in records]),
            text=np.ones((2, 768), dtype=np.float32),
            audio=np.ones((2, 1024), dtype=np.float32),
            vision=np.ones((2, 512), dtype=np.float32),
            modality_mask=np.ones((2, 3), dtype=np.bool_),
        ),
    )
    monkeypatch.setattr(cli, "read_manifest", lambda _path: records)
    output = tmp_path / "reports" / "stats.json"

    result = cli.main(
        [
            "feature-stats",
            "--manifest",
            str(tmp_path / "manifest.jsonl"),
            "--features",
            str(store.root),
            "--dataset",
            "emotiontalk",
            "--split",
            "train",
            "--output",
            str(output),
        ]
    )

    assert result == 0
    assert json.loads(output.read_text(encoding="utf-8"))["sample_count"] == 2
    assert json.loads(capsys.readouterr().out)["shard_count"] == 1


def test_overfit_smoke_command_writes_report(tmp_path, monkeypatch, capsys):
    captured = {}

    def fake_run(records, store, **kwargs):
        captured["records"] = records
        captured["store"] = store
        captured.update(kwargs)
        return {
            "dataset": kwargs["dataset"],
            "split": kwargs["split"],
            "all_passed": True,
            "modalities": {"text": {"passed": True}},
        }

    monkeypatch.setattr(cli, "read_manifest", lambda _path: make_cli_records(2))
    monkeypatch.setattr(cli, "run_unimodal_overfit_smoke", fake_run, raising=False)
    output = tmp_path / "reports" / "overfit.json"

    result = cli.main(
        [
            "overfit-smoke",
            "--manifest",
            str(tmp_path / "manifest.jsonl"),
            "--features",
            str(tmp_path / "features"),
            "--dataset",
            "emotiontalk",
            "--split",
            "train",
            "--output",
            str(output),
            "--device",
            "cpu",
        ]
    )

    assert result == 0
    assert captured["modalities"] == ("text", "audio", "vision")
    assert json.loads(output.read_text(encoding="utf-8"))["all_passed"] is True
    assert json.loads(capsys.readouterr().out)["all_passed"] is True


def test_official_emotiontalk_command_requires_published_csv_inputs():
    parser = build_parser()
    args = parser.parse_args(
        [
            "prepare-emotiontalk-official",
            "--labels-csv",
            "mm.csv",
            "--transcriptions-csv",
            "transcription.csv",
            "--media-root",
            "emotiontalk",
            "--output",
            "emotiontalk.jsonl",
        ]
    )
    assert args.labels_csv == "mm.csv"
    assert args.transcriptions_csv == "transcription.csv"


def test_asr_command_filters_dataset_split_and_writes_incrementally(tmp_path, monkeypatch):
    records = [
        *make_cli_records(2, split="train"),
        *make_cli_records(3, split="test"),
    ]
    captured = {}

    class FakeTranscriber:
        def __init__(self, *, device):
            captured["device"] = device

    def fake_incremental(selected, transcriber, output):
        captured["records"] = list(selected)
        captured["transcriber"] = transcriber
        captured["output"] = output
        return list(selected)

    monkeypatch.setattr(cli, "read_manifest", lambda _path: records)
    monkeypatch.setattr(cli, "FasterWhisperTranscriber", FakeTranscriber)
    monkeypatch.setattr(
        cli,
        "write_asr_manifest_incrementally",
        fake_incremental,
        raising=False,
    )

    result = cli.main(
        [
            "asr-manifest",
            "--manifest",
            str(tmp_path / "all.jsonl"),
            "--output",
            str(tmp_path / "asr-test.jsonl"),
            "--dataset",
            "emotiontalk",
            "--split",
            "test",
            "--device",
            "cuda",
        ]
    )

    assert result == 0
    assert len(captured["records"]) == 3
    assert {str(record.split) for record in captured["records"]} == {"test"}
    assert captured["device"] == "cuda"
    assert captured["output"] == str(tmp_path / "asr-test.jsonl")


def test_asr_command_can_keep_original_text_and_write_error_log(tmp_path, monkeypatch):
    records = make_cli_records(1, split="test")
    captured = {}

    class FakeTranscriber:
        def __init__(self, *, device):
            captured["device"] = device

    def fake_incremental(
        selected,
        transcriber,
        output,
        *,
        keep_original_on_error,
        error_path,
    ):
        captured["keep_original_on_error"] = keep_original_on_error
        captured["error_path"] = error_path
        return list(selected)

    monkeypatch.setattr(cli, "read_manifest", lambda _path: records)
    monkeypatch.setattr(cli, "FasterWhisperTranscriber", FakeTranscriber)
    monkeypatch.setattr(cli, "write_asr_manifest_incrementally", fake_incremental)

    error_path = tmp_path / "asr-errors.jsonl"
    result = cli.main(
        [
            "asr-manifest",
            "--manifest",
            str(tmp_path / "all.jsonl"),
            "--output",
            str(tmp_path / "asr-test.jsonl"),
            "--split",
            "test",
            "--device",
            "cuda",
            "--keep-original-on-error",
            "--error-log",
            str(error_path),
        ]
    )

    assert result == 0
    assert captured["keep_original_on_error"] is True
    assert captured["error_path"] == str(error_path)


def test_train_command_defaults_match_the_approved_plan():
    parser = build_parser()
    args = parser.parse_args(
        [
            "train",
            "--manifest",
            "manifest.jsonl",
            "--features",
            "features",
            "--output",
            "results",
        ]
    )
    assert args.model == "lagf"
    assert args.max_epochs == 50
    assert args.min_epochs == 15
    assert args.patience == 7
    assert args.learning_rate == 1e-4
    assert args.weight_decay == 1e-2
    assert args.training_scope == "joint"
    assert args.skip_test is False
    assert args.augmentation_manifest == []
    assert args.augmentation_features == []
    assert args.no_quality is False


def test_sample_corruption_manifest_parser_defaults():
    args = build_parser().parse_args(
        [
            "sample-corruption-manifest",
            "--manifest",
            "all.jsonl",
            "--output-manifest",
            "selected.jsonl",
            "--base-features",
            "features",
            "--output-features",
            "selected-features",
        ]
    )

    assert args.fraction == 0.1
    assert args.seed == 42
    assert args.dataset is None


def test_sample_corruption_manifest_can_be_scoped_to_one_dataset():
    args = build_parser().parse_args(
        [
            "sample-corruption-manifest",
            "--manifest",
            "all.jsonl",
            "--output-manifest",
            "selected.jsonl",
            "--base-features",
            "features",
            "--output-features",
            "selected-features",
            "--dataset",
            "emotiontalk",
        ]
    )

    assert args.dataset == "emotiontalk"


def test_attach_quality_parser_supports_resumable_shard_ranges():
    args = build_parser().parse_args(
        [
            "attach-quality",
            "--manifest",
            "all.jsonl",
            "--base-features",
            "v1",
            "--output-features",
            "v2",
            "--yunet-model",
            "yunet.onnx",
            "--dataset",
            "meld",
            "--split",
            "train",
            "--start-shard",
            "0",
            "--end-shard",
            "100",
        ]
    )

    assert args.workers == 4
    assert (args.start_shard, args.end_shard) == (0, 100)


def test_runtime_devices_keep_whisper_on_cpu_and_torch_extractors_on_mps():
    torch = __import__("torch")
    assert runtime_devices(torch.device("mps")) == ("mps", "cpu")
    assert runtime_devices(torch.device("cuda")) == ("cuda", "cuda")
    assert runtime_devices(torch.device("cpu")) == ("cpu", "cpu")


def test_analyze_accepts_offline_model_directories():
    args = build_parser().parse_args(
        [
            "analyze",
            "--video",
            "demo.mp4",
            "--checkpoint",
            "best.pt",
            "--yunet-model",
            "yunet.onnx",
            "--text-model",
            "models/xlmr",
            "--audio-model",
            "models/xlsr",
            "--whisper-model",
            "models/whisper-small",
        ]
    )

    assert args.text_model == "models/xlmr"
    assert args.audio_model == "models/xlsr"
    assert args.whisper_model == "models/whisper-small"


def test_analyze_and_serve_accept_a_single_deployment_manifest():
    analyze = build_parser().parse_args(
        [
            "analyze",
            "--video",
            "demo.mp4",
            "--deployment",
            "configs/deployment-v2.json",
        ]
    )
    serve = build_parser().parse_args(
        [
            "serve",
            "--deployment",
            "configs/deployment-v2.json",
        ]
    )
    doctor = build_parser().parse_args(
        [
            "doctor",
            "--deployment",
            "configs/deployment-v2.json",
            "--offline",
        ]
    )

    assert analyze.deployment == "configs/deployment-v2.json"
    assert analyze.checkpoint is None
    assert serve.deployment == "configs/deployment-v2.json"
    assert doctor.offline is True


def test_verify_evidence_command_returns_nonzero_after_artifact_changes(tmp_path, capsys):
    artifact = tmp_path / "result.json"
    artifact.write_text('{"ok": true}\n', encoding="utf-8")
    manifest = tmp_path / "evidence.sha256"
    write_sha256_manifest(
        destination=manifest,
        root=tmp_path,
        inputs=[artifact],
    )

    assert cli.main(["verify-evidence", "--manifest", str(manifest), "--root", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True

    artifact.write_text('{"ok": false}\n', encoding="utf-8")
    assert cli.main(["verify-evidence", "--manifest", str(manifest), "--root", str(tmp_path)]) == 1
    assert json.loads(capsys.readouterr().out)["mismatched"] == ["result.json"]


def test_doctor_reports_missing_deployment_artifacts_without_loading_model(tmp_path, capsys):
    deployment = tmp_path / "deployment.json"
    deployment.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_version": "v2_quality_lagf",
                "architecture": "QualityAwareLanguageGatedFusion",
                "seed": 42,
                "labels": list(EMOTION_LABELS),
                "checkpoint": {"path": "missing.pt", "sha256": "0" * 64},
                "yunet": {"path": "missing.onnx", "sha256": "0" * 64},
                "encoders": {
                    name: {
                        "identifier": name,
                        "revision": "revision",
                        "local_path": f"models/{name}",
                    }
                    for name in ("text", "audio", "vision", "asr")
                },
                "calibration": None,
                "runtime": {"minimum_free_bytes": 0},
                "provenance": {},
            }
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(
        [
            "doctor",
            "--deployment",
            str(deployment),
            "--artifact-root",
            str(tmp_path),
        ]
    )

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert report["ok"] is False
    assert "checkpoint file is missing" in report["errors"]


def test_feature_command_accepts_pre_encoder_robustness_conditions():
    parser = build_parser()
    args = parser.parse_args(
        [
            "extract-features",
            "--manifest",
            "manifest.jsonl",
            "--features",
            "features-snr10-drop25",
            "--yunet-model",
            "yunet.onnx",
            "--audio-snr",
            "10",
            "--frame-drop",
            "0.25",
        ]
    )
    assert args.audio_snr == 10.0
    assert args.frame_drop == 0.25


def test_feature_command_accepts_single_modality_replacement():
    args = build_parser().parse_args(
        [
            "extract-features",
            "--manifest",
            "manifest.jsonl",
            "--features",
            "features-snr10",
            "--base-features",
            "features-clean",
            "--yunet-model",
            "yunet.onnx",
            "--mode",
            "parallel",
            "--only-modality",
            "audio",
            "--condition-name",
            "audio_snr_10db",
        ]
    )

    assert args.only_modality == "audio"
    assert args.base_features == "features-clean"
    assert args.condition_name == "audio_snr_10db"


def test_evaluate_command_accepts_two_missing_modalities():
    args = build_parser().parse_args(
        [
            "evaluate",
            "--manifest",
            "manifest.jsonl",
            "--features",
            "features",
            "--checkpoint",
            "best.pt",
            "--output",
            "missing-text-vision.json",
            "--missing",
            "text",
            "--missing",
            "vision",
        ]
    )

    assert args.missing == ["text", "vision"]


def test_evaluate_command_forwards_robustness_condition(tmp_path, monkeypatch):
    captured = {}

    def fake_evaluate(**kwargs):
        captured.update(kwargs)
        return tmp_path / "result.json"

    monkeypatch.setattr(cli, "evaluate_checkpoint", fake_evaluate)

    result = cli.main(
        [
            "evaluate",
            "--manifest",
            "asr-test.jsonl",
            "--features",
            "features-asr",
            "--checkpoint",
            "best.pt",
            "--output",
            str(tmp_path / "result.json"),
            "--condition-name",
            "whisper_text",
            "--device",
            "cpu",
        ]
    )

    assert result == 0
    assert captured["condition_name"] == "whisper_text"


def test_parallel_feature_defaults_target_dual_t4():
    parser = build_parser()
    args = parser.parse_args(
        [
            "extract-features",
            "--manifest",
            "manifest.jsonl",
            "--features",
            "features",
            "--yunet-model",
            "yunet.onnx",
            "--mode",
            "parallel",
        ]
    )
    assert args.text_audio_device == "cuda:0"
    assert args.vision_device == "cuda:1"
    assert (
        args.text_batch_size,
        args.audio_batch_size,
        args.vision_batch_size,
    ) == (64, 8, 8)
    assert (
        args.audio_workers,
        args.vision_workers,
        args.queue_capacity,
    ) == (4, 4, 8)


def test_extract_parser_accepts_shard_range():
    args = build_parser().parse_args(
        [
            "extract-features",
            "--manifest",
            "manifest.jsonl",
            "--features",
            "features",
            "--yunet-model",
            "yunet.onnx",
            "--dataset",
            "emotiontalk",
            "--split",
            "train",
            "--mode",
            "parallel",
            "--start-shard",
            "120",
            "--end-shard",
            "240",
        ]
    )

    assert (args.start_shard, args.end_shard) == (120, 240)


def test_parallel_range_routes_official_slice_and_offset(tmp_path, monkeypatch):
    records = make_cli_records(40)
    captured = {}

    class FakeRunner:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self, selected, store):
            captured["records"] = selected
            captured["store"] = store
            return []

    monkeypatch.setattr(cli, "read_manifest", lambda _path: records)
    monkeypatch.setattr(cli.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(cli, "ParallelFeatureExtractionRunner", FakeRunner)

    result = cli.main(
        [
            "extract-features",
            "--manifest",
            str(tmp_path / "manifest.jsonl"),
            "--features",
            str(tmp_path / "features"),
            "--yunet-model",
            str(tmp_path / "yunet.onnx"),
            "--dataset",
            "emotiontalk",
            "--split",
            "train",
            "--mode",
            "parallel",
            "--shard-size",
            "16",
            "--start-shard",
            "1",
            "--end-shard",
            "3",
        ]
    )

    assert result == 0
    assert captured["records"] == records[16:40]
    assert captured["config"].shard_index_offset == 1


@pytest.mark.parametrize(
    "extra",
    [
        ["--mode", "serial", "--start-shard", "0", "--end-shard", "1"],
        ["--mode", "parallel", "--start-shard", "0"],
    ],
)
def test_invalid_range_requests_fail_before_model_construction(tmp_path, monkeypatch, extra):
    monkeypatch.setattr(cli, "read_manifest", lambda _path: make_cli_records(16))
    monkeypatch.setattr(cli.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        cli,
        "TextFeatureExtractor",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("model must not be constructed")),
    )
    args = [
        "extract-features",
        "--manifest",
        str(tmp_path / "manifest.jsonl"),
        "--features",
        str(tmp_path / "features"),
        "--yunet-model",
        str(tmp_path / "yunet.onnx"),
        "--dataset",
        "emotiontalk",
        "--split",
        "train",
        *extra,
    ]

    with pytest.raises((ValueError, SystemExit)):
        cli.main(args)


def test_verify_features_command_prints_json_and_writes_completion(tmp_path, monkeypatch, capsys):
    records = make_cli_records(32)
    store = FeatureStore(tmp_path / "features")
    for shard_index, chunk in enumerate((records[:16], records[16:])):
        rows = len(chunk)
        store.write(
            "emotiontalk",
            "train",
            shard_index,
            FeatureShard(
                sample_ids=np.asarray([record.sample_id for record in chunk]),
                text=np.ones((rows, 768), dtype=np.float32),
                audio=np.ones((rows, 1024), dtype=np.float32),
                vision=np.ones((rows, 512), dtype=np.float32),
                modality_mask=np.ones((rows, 3), dtype=np.bool_),
            ),
        )
    monkeypatch.setattr(cli, "read_manifest", lambda _path: records)

    result = cli.main(
        [
            "verify-features",
            "--manifest",
            str(tmp_path / "manifest.jsonl"),
            "--features",
            str(store.root),
            "--dataset",
            "emotiontalk",
            "--split",
            "train",
            "--shard-size",
            "16",
            "--start-shard",
            "0",
            "--end-shard",
            "2",
            "--write-completion",
        ]
    )

    assert result == 0
    assert json.loads(capsys.readouterr().out)["is_valid"] is True
    assert (store.root / "ranges" / "range-00000-00002.json").is_file()


def test_serial_feature_mode_remains_default():
    parser = build_parser()
    args = parser.parse_args(
        [
            "extract-features",
            "--manifest",
            "manifest.jsonl",
            "--features",
            "features",
            "--yunet-model",
            "yunet.onnx",
        ]
    )
    assert args.mode == "serial"


def test_parallel_feature_command_routes_to_parallel_runner(tmp_path, monkeypatch):
    record = UtteranceRecord(
        dataset="emotiontalk",
        split="validation",
        dialogue_id="d1",
        utterance_id=0,
        text="line",
        emotion="neutral",
        language="zh",
        start_seconds=0.0,
        end_seconds=1.0,
        video_path=tmp_path / "0.mp4",
    )
    captured = {}

    class FakeRunner:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self, records, store):
            captured["records"] = records
            captured["store"] = store
            return []

    monkeypatch.setattr(cli, "read_manifest", lambda _path: [record])
    monkeypatch.setattr(cli.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        cli,
        "TextFeatureExtractor",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("parallel mode must lazily construct text extractor")
        ),
    )
    monkeypatch.setattr(
        cli,
        "ParallelFeatureExtractionRunner",
        FakeRunner,
        raising=False,
    )

    result = cli.main(
        [
            "extract-features",
            "--manifest",
            str(tmp_path / "manifest.jsonl"),
            "--features",
            str(tmp_path / "features"),
            "--staging",
            str(tmp_path / "staging"),
            "--yunet-model",
            str(tmp_path / "yunet.onnx"),
            "--dataset",
            "emotiontalk",
            "--split",
            "validation",
            "--mode",
            "parallel",
            "--shard-size",
            "16",
        ]
    )

    assert result == 0
    assert captured["config"].shard_size == 16
    assert captured["staging_root"] == tmp_path / "staging"
    assert captured["records"] == [record]
    assert captured["audio_executor_factory"].keywords["mp_context"].get_start_method() == "spawn"
    assert captured["vision_executor_factory"].keywords["mp_context"].get_start_method() == "spawn"


def test_parallel_audio_replacement_does_not_require_unused_vision_gpu(tmp_path, monkeypatch):
    records = make_cli_records(2, split="test")
    captured = {}

    class FakeRunner:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self, selected, store):
            captured["records"] = selected
            captured["store"] = store
            return []

    def fake_seed(**kwargs):
        captured.setdefault("seed_calls", []).append(kwargs)
        return []

    monkeypatch.setattr(cli, "read_manifest", lambda _path: records)
    monkeypatch.setattr(cli.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(cli, "ParallelFeatureExtractionRunner", FakeRunner)
    monkeypatch.setattr(cli, "seed_staging_from_base_shard", fake_seed, raising=False)
    result = cli.main(
        [
            "extract-features",
            "--manifest",
            str(tmp_path / "manifest.jsonl"),
            "--features",
            str(tmp_path / "snr10"),
            "--base-features",
            str(tmp_path / "clean"),
            "--yunet-model",
            str(tmp_path / "yunet.onnx"),
            "--dataset",
            "emotiontalk",
            "--split",
            "test",
            "--mode",
            "parallel",
            "--only-modality",
            "audio",
            "--condition-name",
            "audio_snr_10db",
            "--audio-snr",
            "10",
        ]
    )

    assert result == 0
    assert captured["records"] == records
    assert captured["store"].root == tmp_path / "snr10"
    assert len(captured["seed_calls"]) == 1
    assert captured["seed_calls"][0]["base_store"].root == tmp_path / "clean"
    assert captured["seed_calls"][0]["recompute_modality"] == "audio"
    provenance_path = tmp_path / "snr10" / "emotiontalk" / "test" / "condition.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert provenance["condition"] == "audio_snr_10db"
    assert provenance["dataset"] == "emotiontalk"
    assert provenance["split"] == "test"
    assert provenance["audio_snr"] == 10.0
