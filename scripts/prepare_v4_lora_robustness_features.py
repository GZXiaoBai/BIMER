#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bimer.feature_store import FeatureStore
from bimer.lora_text_encoder import load_adapted_text_extractor
from bimer.manifest import read_manifest
from bimer.text_adaptation import compose_feature_stores, rewrite_text_feature_store

PARTITIONS = (("meld", "test"), ("emotiontalk", "test"))
VIEWS = ("audio_snr_10db", "video_frame_drop_50pct", "whisper_text")


def _directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(file_path.relative_to(path)).encode())
        digest.update(file_path.read_bytes())
    return digest.hexdigest()


def _device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def prepare(args: argparse.Namespace) -> dict[str, object]:
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if selection.get("state") != "frozen" or selection.get("version") != "v4":
        raise RuntimeError("V4 selection is not frozen")
    config = selection["candidate_config"]
    required = (
        "feature_root",
        "adapter_path",
        "adapter_base_model",
        "adapter_sha256",
    )
    if any(not config.get(field) for field in required):
        raise RuntimeError("frozen selection does not contain a LoRA adapter")
    payload = {
        "standard_features": str(config["feature_root"]),
        "adapter_path": str(config["adapter_path"]),
        "adapter_base_model": str(config["adapter_base_model"]),
        "adapter_sha256": str(config["adapter_sha256"]),
        "robustness_features": str(args.robustness_features),
        "whisper_manifest": str(args.whisper_manifest),
        "output": str(args.output),
        "views": list(VIEWS),
    }
    if args.dry_run:
        return payload

    adapter = Path(str(config["adapter_path"]))
    actual_hash = _directory_sha256(adapter)
    if actual_hash != config["adapter_sha256"]:
        raise ValueError("LoRA adapter SHA-256 does not match the frozen selection")
    standard = FeatureStore(str(config["feature_root"]))
    robustness = Path(args.robustness_features)
    compose_feature_stores(
        standard,
        FeatureStore(args.output / "audio_snr_10db"),
        replacements={"audio": FeatureStore(robustness / "audio_snr_10db")},
        partitions=PARTITIONS,
    )
    compose_feature_stores(
        standard,
        FeatureStore(args.output / "video_frame_drop_50pct"),
        replacements={"vision": FeatureStore(robustness / "video_frame_drop_50pct")},
        partitions=PARTITIONS,
    )

    records = [
        record for record in read_manifest(args.whisper_manifest) if str(record.split) == "test"
    ]
    sample_ids = [record.sample_id for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Whisper manifest sample IDs must be unique")
    extractor = load_adapted_text_extractor(
        str(config["adapter_base_model"]),
        adapter,
        device=_device(args.device),
        local_files_only=args.local_files_only,
    )
    encoded = extractor.encode(
        [record.text for record in records],
        batch_size=args.batch_size,
    )
    if encoded.shape != (len(records), 768):
        raise ValueError("LoRA Whisper text features must remain 768-dimensional")
    replacements = dict(zip(sample_ids, encoded, strict=True))
    whisper_source = FeatureStore(robustness / "whisper_text")
    source_ids = {
        sample_id
        for dataset, split in PARTITIONS
        for shard in whisper_source.read_all(dataset, split)
        for sample_id in np.asarray(shard.sample_ids).astype(str).tolist()
    }
    if source_ids != set(sample_ids):
        raise ValueError("Whisper feature store IDs do not match the Whisper manifest")
    rewrite_text_feature_store(
        whisper_source,
        FeatureStore(args.output / "whisper_text"),
        replacements=replacements,
        partitions=PARTITIONS,
        expected_dim=768,
    )
    payload.update(
        {
            "device": _device(args.device),
            "records": len(records),
            "sample_ids_preserved": True,
        }
    )
    _atomic_json(args.output / "LORA_ROBUSTNESS_FEATURES_READY.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--whisper-manifest", type=Path, required=True)
    parser.add_argument("--robustness-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
