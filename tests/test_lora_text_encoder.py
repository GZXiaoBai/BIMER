from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from bimer.lora_text_encoder import AdaptedTextFeatureExtractor, LoraTextClassifier


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10, 4)

    def forward(self, input_ids, attention_mask):
        del attention_mask
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        del kwargs
        rows = [[len(text) % 5 + 1, 2] for text in texts]
        return {
            "input_ids": torch.tensor(rows),
            "attention_mask": torch.ones(len(rows), 2, dtype=torch.long),
        }


def test_lora_text_classifier_returns_logits_and_mean_pooled_embeddings():
    model = LoraTextClassifier(FakeEncoder(), hidden_size=4, num_classes=3)
    tokens = {
        "input_ids": torch.tensor([[1, 2], [3, 4]]),
        "attention_mask": torch.tensor([[1, 1], [1, 0]]),
    }

    logits, embeddings = model(**tokens)

    assert logits.shape == (2, 3)
    assert embeddings.shape == (2, 4)
    expected_second = model.encoder.embedding(torch.tensor([3]))
    torch.testing.assert_close(embeddings[1], expected_second[0])


def test_adapted_text_extractor_keeps_768_contract_configurable_for_tests():
    extractor = AdaptedTextFeatureExtractor(
        tokenizer=FakeTokenizer(),
        encoder=FakeEncoder(),
        device="cpu",
        output_dim=4,
    )

    encoded = extractor.encode(["hello", "世界"], batch_size=1)

    assert encoded.shape == (2, 4)
    assert encoded.dtype == np.float32
    assert np.isfinite(encoded).all()
