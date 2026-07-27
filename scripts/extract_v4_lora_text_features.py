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
from bimer.text_adaptation import rewrite_text_feature_store

EXPECTED_DIM = 768


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


def extract(args: argparse.Namespace) -> dict[str, object]:
    dry_run = {
        "manifest": str(args.manifest),
        "source_features": str(args.source_features),
        "output_features": str(args.output_features),
        "base_model": args.base_model,
        "adapter": str(args.adapter),
        "expected_dim": EXPECTED_DIM,
    }
    if args.dry_run:
        return dry_run

    records = read_manifest(args.manifest)
    sample_ids = [record.sample_id for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("manifest sample IDs must be unique")
    extractor = load_adapted_text_extractor(
        args.base_model,
        args.adapter,
        device=_device(args.device),
        local_files_only=args.local_files_only,
    )
    encoded = extractor.encode(
        [record.text for record in records],
        batch_size=args.batch_size,
    )
    if encoded.shape != (len(records), EXPECTED_DIM):
        raise ValueError(f"adapted text features must have shape ({len(records)}, {EXPECTED_DIM})")
    replacements = dict(zip(sample_ids, encoded, strict=True))
    source = FeatureStore(args.source_features)
    destination = FeatureStore(args.output_features)
    partitions = sorted({(record.dataset.lower(), str(record.split)) for record in records})
    source_ids: set[str] = set()
    for dataset, split in partitions:
        paths = source.paths(dataset, split)
        if not paths:
            raise ValueError(f"source features missing partition {dataset}/{split}")
        for path in paths:
            source_ids.update(np.asarray(source.read(path).sample_ids).astype(str).tolist())
    if source_ids != set(sample_ids):
        missing = sorted(set(sample_ids) - source_ids)[:5]
        unexpected = sorted(source_ids - set(sample_ids))[:5]
        raise ValueError(
            f"source feature IDs do not match manifest; missing={missing}, unexpected={unexpected}"
        )
    written = rewrite_text_feature_store(
        source,
        destination,
        replacements=replacements,
        partitions=partitions,
        expected_dim=EXPECTED_DIM,
    )
    payload = {
        **dry_run,
        "device": _device(args.device),
        "records": len(records),
        "partitions": [list(partition) for partition in partitions],
        "feature_shards": len(written),
        "adapter_sha256": _directory_sha256(Path(args.adapter)),
        "sample_ids_preserved": True,
        "official_splits_preserved": True,
    }
    _atomic_json(Path(args.output_features) / "TEXT_FEATURES_READY.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-features", type=Path, required=True)
    parser.add_argument("--output-features", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(extract(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
