import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "scripts" / "train_v4_text_lora.py"
EXTRACT = ROOT / "scripts" / "extract_v4_lora_text_features.py"


def test_lora_training_dry_run_uses_frozen_configuration(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(TRAIN),
            "--manifest",
            "manifest.jsonl",
            "--base-model",
            "xlm-roberta-base",
            "--output",
            str(tmp_path / "adapter"),
            "--learning-rate",
            "0.0001",
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["rank"] == 8
    assert payload["alpha"] == 16
    assert payload["max_epochs"] == 5
    assert payload["max_length"] == 128
    assert payload["contrastive_weight"] == 0.1


def test_lora_feature_extraction_dry_run_records_source_and_destination(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(EXTRACT),
            "--manifest",
            "manifest.jsonl",
            "--source-features",
            "source",
            "--output-features",
            "destination",
            "--base-model",
            "xlm-roberta-base",
            "--adapter",
            str(tmp_path / "adapter"),
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["source_features"] == "source"
    assert payload["output_features"] == "destination"
    assert payload["expected_dim"] == 768
