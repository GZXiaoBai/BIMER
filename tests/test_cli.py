import argparse

import bimer.cli as cli
from bimer.cli import build_parser
from bimer.schema import UtteranceRecord


def test_cli_exposes_all_reproducible_workflow_commands():
    parser = build_parser()
    expected = {
        "prepare-meld",
        "prepare-emotiontalk",
        "prepare-emotiontalk-official",
        "asr-manifest",
        "validate",
        "extract-features",
        "train",
        "evaluate",
        "analyze",
        "serve",
    }
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    assert set(subparsers.choices) == expected


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
    assert args.patience == 7
    assert args.learning_rate == 1e-4
    assert args.weight_decay == 1e-2
    assert args.training_scope == "joint"


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
    assert (
        captured["audio_executor_factory"]
        .keywords["mp_context"]
        .get_start_method()
        == "spawn"
    )
    assert (
        captured["vision_executor_factory"]
        .keywords["mp_context"]
        .get_start_method()
        == "spawn"
    )
