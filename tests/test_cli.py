import argparse

from bimer.cli import build_parser


def test_cli_exposes_all_reproducible_workflow_commands():
    parser = build_parser()
    expected = {
        "prepare-meld",
        "prepare-emotiontalk",
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
