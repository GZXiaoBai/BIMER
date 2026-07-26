#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bimer.labels import EMOTION_LABELS, emotion_index
from bimer.lora_text_encoder import build_lora_text_classifier
from bimer.losses import sqrt_inverse_class_weights
from bimer.manifest import read_manifest
from bimer.metrics import classification_metrics
from bimer.text_adaptation import (
    BalancedTextSampler,
    LoraTextAdaptationConfig,
    SupervisedContrastiveLoss,
)


class _TextRows(Dataset):
    def __init__(self, records) -> None:
        self.records = tuple(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        return record.text, emotion_index(str(record.emotion)), record.dataset.lower()


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


def _device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _collate(tokenizer, max_length: int):
    def collate(rows):
        texts, labels, datasets = zip(*rows, strict=True)
        tokens = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return tokens, torch.tensor(labels, dtype=torch.long), tuple(datasets)

    return collate


@torch.inference_mode()
def _evaluate(model, loader, *, device: torch.device) -> dict[str, object]:
    model.eval()
    truth: list[int] = []
    prediction: list[int] = []
    for tokens, labels, _ in loader:
        tokens = {name: values.to(device) for name, values in tokens.items()}
        logits, _ = model(**tokens)
        truth.extend(labels.tolist())
        prediction.extend(logits.argmax(dim=-1).cpu().tolist())
    return classification_metrics(
        np.asarray(truth, dtype=np.int64),
        np.asarray(prediction, dtype=np.int64),
        label_names=EMOTION_LABELS,
    )


def train(args: argparse.Namespace) -> dict[str, object]:
    config = LoraTextAdaptationConfig(learning_rate=args.learning_rate)
    dry_run = {
        **asdict(config),
        "manifest": str(args.manifest),
        "base_model": args.base_model,
        "output": str(args.output),
        "seed": args.seed,
    }
    if args.dry_run:
        return dry_run

    if args.seed != 42:
        raise ValueError("V4 LoRA screening is restricted to seed 42")
    records = read_manifest(args.manifest)
    train_rows = [
        row
        for row in records
        if str(row.split) == "train" and row.dataset.lower() in {"meld", "emotiontalk"}
    ]
    validation_rows = {
        "meld": [
            row
            for row in records
            if row.dataset.lower() == "meld" and str(row.split) in {"dev", "validation", "val"}
        ],
        "emotiontalk": [
            row
            for row in records
            if row.dataset.lower() == "emotiontalk"
            and str(row.split) in {"dev", "validation", "val"}
        ],
    }
    if not train_rows or any(not rows for rows in validation_rows.values()):
        raise ValueError("manifest must contain bilingual train and validation rows")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _device(args.device)
    tokenizer, model = build_lora_text_classifier(
        args.base_model,
        config=config,
        num_classes=len(EMOTION_LABELS),
        local_files_only=args.local_files_only,
    )
    model.to(device)
    collate = _collate(tokenizer, config.max_length)
    sampler = BalancedTextSampler(
        [row.dataset.lower() for row in train_rows],
        seed=args.seed,
    )
    train_loader = DataLoader(
        _TextRows(train_rows),
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate,
    )
    validation_loaders = {
        dataset: DataLoader(
            _TextRows(rows),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate,
        )
        for dataset, rows in validation_rows.items()
    }
    labels = torch.tensor(
        [emotion_index(str(row.emotion)) for row in train_rows],
        dtype=torch.long,
    )
    class_weights = sqrt_inverse_class_weights(
        labels,
        num_classes=len(EMOTION_LABELS),
    ).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=1e-2,
    )
    contrastive = SupervisedContrastiveLoss(temperature=config.temperature)
    output = Path(args.output)
    adapter_directory = output / "adapter"
    best_score = float("-inf")
    history: list[dict[str, object]] = []

    for epoch in range(config.max_epochs):
        sampler.set_epoch(epoch)
        model.train()
        losses: list[float] = []
        for tokens, batch_labels, _ in train_loader:
            tokens = {name: values.to(device) for name, values in tokens.items()}
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, embeddings = model(**tokens)
            classification = torch.nn.functional.cross_entropy(
                logits,
                batch_labels,
                weight=class_weights,
            )
            supervised_contrastive = contrastive(embeddings, batch_labels)
            loss = classification + config.contrastive_weight * supervised_contrastive
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite LoRA training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        validation = {
            dataset: _evaluate(model, loader, device=device)
            for dataset, loader in validation_loaders.items()
        }
        score = float(
            np.mean(
                [
                    0.5 * (metrics["weighted_f1"] + metrics["macro_f1"])
                    for metrics in validation.values()
                ]
            )
        )
        history.append(
            {
                "epoch": epoch + 1,
                "training_loss": float(np.mean(losses)),
                "selection_score": score,
                "validation": validation,
            }
        )
        if score > best_score:
            best_score = score
            adapter_directory.mkdir(parents=True, exist_ok=True)
            model.encoder.save_pretrained(adapter_directory)
            tokenizer.save_pretrained(output / "tokenizer")
            torch.save(
                model.classifier.state_dict(),
                output / "classifier.pt",
            )
            _atomic_json(output / "history.json", history)

    result = {
        **dry_run,
        "device": str(device),
        "best_selection_score": best_score,
        "adapter_path": str(adapter_directory),
        "adapter_sha256": _directory_sha256(adapter_directory),
        "history": history,
        "evidence_scope": "validation_only",
        "test_set_used": False,
    }
    _atomic_json(output / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, choices=(1e-4, 2e-4), required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(train(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
