from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .feature_extractors import mean_pool_hidden
from .text_adaptation import LoraTextAdaptationConfig


class LoraTextClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, *, hidden_size: int, num_classes: int = 7) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        **tokens: Tensor,
    ) -> tuple[Tensor, Tensor]:
        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **tokens,
        )
        embeddings = mean_pool_hidden(output.last_hidden_state, attention_mask)
        return self.classifier(embeddings), embeddings


class AdaptedTextFeatureExtractor:
    def __init__(
        self,
        *,
        tokenizer: Any,
        encoder: nn.Module,
        device: str,
        output_dim: int = 768,
    ) -> None:
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.encoder = encoder.to(self.device).eval()
        self.encoder.requires_grad_(False)
        self.output_dim = output_dim

    @torch.inference_mode()
    def encode(self, texts: Sequence[str], *, batch_size: int = 16) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        outputs = []
        for start in range(0, len(texts), batch_size):
            tokens = self.tokenizer(
                list(texts[start : start + batch_size]),
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            tokens = {name: values.to(self.device) for name, values in tokens.items()}
            hidden = self.encoder(**tokens).last_hidden_state
            pooled = mean_pool_hidden(hidden, tokens["attention_mask"])
            if pooled.shape[1] != self.output_dim:
                raise ValueError(f"adapted text encoder output width must be {self.output_dim}")
            outputs.append(pooled.cpu().numpy().astype(np.float32))
        return (
            np.concatenate(outputs) if outputs else np.empty((0, self.output_dim), dtype=np.float32)
        )


def build_lora_text_classifier(
    base_model: str,
    *,
    config: LoraTextAdaptationConfig,
    num_classes: int = 7,
    local_files_only: bool = False,
) -> tuple[object, LoraTextClassifier]:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install bimer[adaptation] to train the V4 LoRA fallback") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        local_files_only=local_files_only,
    )
    encoder = AutoModel.from_pretrained(
        base_model,
        local_files_only=local_files_only,
    )
    encoder = get_peft_model(
        encoder,
        LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=config.rank,
            lora_alpha=config.alpha,
            lora_dropout=config.dropout,
            target_modules=["query", "value"],
        ),
    )
    hidden_size = int(encoder.config.hidden_size)
    return tokenizer, LoraTextClassifier(
        encoder,
        hidden_size=hidden_size,
        num_classes=num_classes,
    )


def load_adapted_text_extractor(
    base_model: str,
    adapter_path: Path | str,
    *,
    device: str,
    local_files_only: bool = False,
) -> AdaptedTextFeatureExtractor:
    try:
        from peft import PeftModel
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install bimer[adaptation] to load the V4 LoRA fallback") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        local_files_only=local_files_only,
    )
    encoder = AutoModel.from_pretrained(
        base_model,
        local_files_only=local_files_only,
    )
    encoder = PeftModel.from_pretrained(encoder, str(adapter_path))
    return AdaptedTextFeatureExtractor(
        tokenizer=tokenizer,
        encoder=encoder,
        device=device,
        output_dim=int(encoder.config.hidden_size),
    )
