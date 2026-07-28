from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import torch

import bimer.feature_extractors as feature_module
from bimer.feature_extractors import (
    AudioFeatureExtractor,
    TextFeatureExtractor,
    VisionFeatureExtractor,
)


class _FrozenModel:
    def to(self, _device):
        return self

    def eval(self):
        return self

    def requires_grad_(self, _required: bool):
        return self


class _TextModel(_FrozenModel):
    def __call__(self, **tokens):
        batch, length = tokens["attention_mask"].shape
        hidden = torch.arange(batch * length * 4, dtype=torch.float32).reshape(batch, length, 4)
        return SimpleNamespace(last_hidden_state=hidden)


class _AudioModel(_FrozenModel):
    def __call__(self, **inputs):
        batch = inputs["input_values"].shape[0]
        return SimpleNamespace(last_hidden_state=torch.ones(batch, 3, 4))

    def _get_feature_vector_attention_mask(self, length: int, attention_mask):
        return torch.ones(attention_mask.shape[0], length, dtype=torch.long)


def _install_fake_transformers(monkeypatch) -> None:
    module = ModuleType("transformers")

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(_name: str):
            return lambda batch, **_kwargs: {
                "input_ids": torch.ones(len(batch), 2, dtype=torch.long),
                "attention_mask": torch.tensor([[1, 0]] * len(batch)),
            }

    class AutoFeatureExtractor:
        @staticmethod
        def from_pretrained(_name: str):
            def process(batch, **_kwargs):
                width = max(len(item) for item in batch)
                padded = np.stack([np.pad(item, (0, width - len(item))) for item in batch])
                return SimpleNamespace(
                    input_values=torch.as_tensor(padded),
                    attention_mask=torch.ones(len(batch), width, dtype=torch.long),
                )

            return process

    class AutoModel:
        @staticmethod
        def from_pretrained(name: str):
            return _AudioModel() if name == "audio-model" else _TextModel()

    module.AutoTokenizer = AutoTokenizer
    module.AutoFeatureExtractor = AutoFeatureExtractor
    module.AutoModel = AutoModel
    monkeypatch.setitem(sys.modules, "transformers", module)


def test_text_and_audio_extractors_run_with_frozen_transformer_contract(
    monkeypatch,
) -> None:
    _install_fake_transformers(monkeypatch)

    text = TextFeatureExtractor("text-model")
    audio = AudioFeatureExtractor("audio-model")

    assert text.encode(["one", "two"], batch_size=1).shape == (2, 4)
    assert text.encode([]).shape == (0, 768)
    assert audio.encode([np.ones(400, np.float32), np.ones(500, np.float32)]).shape == (
        2,
        4,
    )


def _install_fake_torchvision(monkeypatch) -> None:
    torchvision = ModuleType("torchvision")
    models = ModuleType("torchvision.models")
    video = ModuleType("torchvision.models.video")

    class FakeVideoModel(_FrozenModel):
        fc = None

        def __call__(self, tensor):
            return torch.ones(tensor.shape[0], 512)

        def load_state_dict(self, _state) -> None:
            pass

    class R3D18Weights:
        DEFAULT = object()

    video.R3D_18_Weights = R3D18Weights
    video.r3d_18 = lambda **_kwargs: FakeVideoModel()
    monkeypatch.setitem(sys.modules, "torchvision", torchvision)
    monkeypatch.setitem(sys.modules, "torchvision.models", models)
    monkeypatch.setitem(sys.modules, "torchvision.models.video", video)


def test_vision_extractor_default_model_and_available_unavailable_paths(
    monkeypatch,
) -> None:
    _install_fake_torchvision(monkeypatch)
    extractor = VisionFeatureExtractor(device="cpu")
    clip = np.zeros((16, 8, 8, 3), dtype=np.uint8)

    assert extractor.encode_clips([clip, clip], batch_size=1).shape == (2, 512)

    monkeypatch.setattr(
        feature_module,
        "prepare_video_clip_with_quality",
        lambda *_args, **_kwargs: (clip, False, np.zeros(4, np.float32)),
    )
    missing, available, quality = extractor.encode_video_with_quality(
        "video.mp4",
        face_cropper=object(),
    )
    assert missing.shape == (1, 512)
    assert available is False
    assert not quality.any()

    monkeypatch.setattr(
        feature_module,
        "prepare_video_segment_with_quality",
        lambda *_args, **_kwargs: (clip, True, np.ones(4, np.float32)),
    )
    present, available, quality = extractor.encode_video_segment_with_quality(
        "video.mp4",
        start_seconds=0.0,
        end_seconds=1.0,
        face_cropper=object(),
    )
    assert present.shape == (1, 512)
    assert available is True
    assert quality.tolist() == [1.0] * 4
