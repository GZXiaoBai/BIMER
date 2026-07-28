from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, Sequence

import torch
from torch import Tensor
from torch.utils.data import Sampler

from .batching import DialogueExample, MultimodalBatch, collate_dialogues

ModalityName = Literal["text", "audio", "vision"]
MODALITY_INDEX: dict[ModalityName, int] = {
    "text": 0,
    "audio": 1,
    "vision": 2,
}


@dataclass(frozen=True, slots=True)
class CorruptionPair:
    clean: DialogueExample
    corrupted: DialogueExample
    corrupted_modality: ModalityName
    severity: float = 1.0

    def __post_init__(self) -> None:
        if self.corrupted_modality not in MODALITY_INDEX:
            raise ValueError("corrupted_modality must be text, audio, or vision")
        if self.clean.dataset != self.corrupted.dataset:
            raise ValueError("clean and corrupted examples must use the same dataset")
        if self.clean.sample_ids != self.corrupted.sample_ids:
            raise ValueError("clean and corrupted examples must have identical sample order")
        if self.clean.language_id != self.corrupted.language_id:
            raise ValueError("clean and corrupted examples must use the same language")
        if not torch.equal(
            torch.from_numpy(self.clean.labels),
            torch.from_numpy(self.corrupted.labels),
        ):
            raise ValueError("clean and corrupted examples must have identical labels")


@dataclass(slots=True)
class PairedMultimodalBatch:
    clean: MultimodalBatch
    corrupted: MultimodalBatch
    corrupted_modality: Tensor
    severity: Tensor

    def to(self, device: torch.device) -> "PairedMultimodalBatch":
        return PairedMultimodalBatch(
            clean=self.clean.to(device),
            corrupted=self.corrupted.to(device),
            corrupted_modality=self.corrupted_modality.to(device),
            severity=self.severity.to(device),
        )


class BalancedCorruptionPairSampler(Sampler[int]):
    """Deterministically reshuffle and 1:1 balance bilingual corruption pairs."""

    def __init__(self, pairs: Sequence[CorruptionPair], *, seed: int = 42) -> None:
        self.pairs = pairs
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        groups: dict[str, list[int]] = {"meld": [], "emotiontalk": []}
        for index, pair in enumerate(self.pairs):
            groups.setdefault(pair.clean.dataset, []).append(index)
        generator = random.Random(self.seed + self.epoch)
        for indices in groups.values():
            generator.shuffle(indices)
        if not groups["meld"] or not groups["emotiontalk"]:
            indices = list(range(len(self.pairs)))
            generator.shuffle(indices)
            return iter(indices)
        target = max(len(groups["meld"]), len(groups["emotiontalk"]))
        balanced = [
            groups[dataset][offset % len(groups[dataset])]
            for offset in range(target)
            for dataset in ("meld", "emotiontalk")
        ]
        return iter(balanced)

    def __len__(self) -> int:
        counts = {
            dataset: sum(pair.clean.dataset == dataset for pair in self.pairs)
            for dataset in ("meld", "emotiontalk")
        }
        if counts["meld"] and counts["emotiontalk"]:
            return 2 * max(counts.values())
        return len(self.pairs)


def build_corruption_pairs(
    clean_examples: Sequence[DialogueExample],
    corrupted_examples: Sequence[DialogueExample],
    *,
    corrupted_modality: ModalityName,
    severity: float = 1.0,
) -> tuple[CorruptionPair, ...]:
    clean_by_samples: dict[tuple[str, ...], DialogueExample] = {}
    for example in clean_examples:
        if example.sample_ids in clean_by_samples:
            raise ValueError("clean examples contain duplicate sample windows")
        clean_by_samples[example.sample_ids] = example

    pairs: list[CorruptionPair] = []
    seen: set[tuple[str, ...]] = set()
    for corrupted in corrupted_examples:
        clean = clean_by_samples.get(corrupted.sample_ids)
        if clean is None:
            raise ValueError("corrupted example has no matching clean window")
        if corrupted.sample_ids in seen:
            raise ValueError("corrupted examples contain duplicate sample windows")
        seen.add(corrupted.sample_ids)
        pairs.append(
            CorruptionPair(
                clean=clean,
                corrupted=corrupted,
                corrupted_modality=corrupted_modality,
                severity=float(severity),
            )
        )
    if not pairs:
        raise ValueError("at least one corruption pair is required")
    return tuple(pairs)


def collate_corruption_pairs(
    pairs: Sequence[CorruptionPair],
) -> PairedMultimodalBatch:
    if not pairs:
        raise ValueError("cannot collate an empty corruption-pair batch")
    clean = collate_dialogues([pair.clean for pair in pairs])
    corrupted = collate_dialogues([pair.corrupted for pair in pairs])
    if clean.sample_ids != corrupted.sample_ids:
        raise ValueError("paired batch lost clean/corrupted sample alignment")
    return PairedMultimodalBatch(
        clean=clean,
        corrupted=corrupted,
        corrupted_modality=torch.tensor(
            [MODALITY_INDEX[pair.corrupted_modality] for pair in pairs],
            dtype=torch.long,
        ),
        severity=torch.tensor([pair.severity for pair in pairs], dtype=torch.float32),
    )


def gate_ranking_loss(
    clean_gates: Tensor,
    corrupted_gates: Tensor,
    *,
    clean_modality_mask: Tensor,
    corrupted_modality_mask: Tensor,
    attention_mask: Tensor,
    corrupted_modality: Tensor,
    margin: float = 0.10,
) -> Tensor:
    if clean_gates.shape != corrupted_gates.shape:
        raise ValueError("clean and corrupted gates must have the same shape")
    if clean_gates.ndim != 3 or clean_gates.shape[-1] != 3:
        raise ValueError("gates must have shape [batch, sequence, 3]")
    if clean_modality_mask.shape != clean_gates.shape:
        raise ValueError("clean_modality_mask must match gates")
    if corrupted_modality_mask.shape != clean_gates.shape:
        raise ValueError("corrupted_modality_mask must match gates")
    if attention_mask.shape != clean_gates.shape[:2]:
        raise ValueError("attention_mask must have shape [batch, sequence]")
    if corrupted_modality.shape != (clean_gates.shape[0],):
        raise ValueError("corrupted_modality must have shape [batch]")
    if margin < 0:
        raise ValueError("margin must be non-negative")
    if bool(((corrupted_modality < 0) | (corrupted_modality > 2)).any()):
        raise ValueError("corrupted_modality values must be 0, 1, or 2")

    gather_index = corrupted_modality[:, None, None].expand(-1, clean_gates.shape[1], 1)
    clean_selected = clean_gates.gather(-1, gather_index).squeeze(-1)
    corrupted_selected = corrupted_gates.gather(-1, gather_index).squeeze(-1)
    clean_available = clean_modality_mask.gather(-1, gather_index).squeeze(-1)
    corrupted_available = corrupted_modality_mask.gather(-1, gather_index).squeeze(-1)
    valid = attention_mask.bool() & clean_available.bool() & corrupted_available.bool()
    if not bool(valid.any()):
        return clean_gates.sum() * 0.0 + corrupted_gates.sum() * 0.0
    violations = torch.relu(
        torch.as_tensor(margin, dtype=clean_gates.dtype, device=clean_gates.device)
        - (clean_selected - corrupted_selected)
    )
    return violations[valid].mean()
