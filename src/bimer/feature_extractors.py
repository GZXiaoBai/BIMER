from __future__ import annotations

import io
import subprocess
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn

from .robustness import drop_video_frames


def mean_pool_hidden(hidden: Tensor, attention_mask: Tensor) -> Tensor:
    weights = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def uniform_frame_indices(total_frames: int, requested: int = 16) -> np.ndarray:
    if total_frames <= 0 or requested <= 0:
        raise ValueError("total_frames and requested must be positive")
    return np.rint(np.linspace(0, total_frames - 1, requested)).astype(np.int64)


def vision_modality_available(
    detected_faces: np.ndarray | Sequence[bool], *, minimum_faces: int = 4
) -> bool:
    return int(np.asarray(detected_faces, dtype=np.bool_).sum()) >= minimum_faces


class TextFeatureExtractor:
    output_dim = 768

    def __init__(self, model_name: str = "xlm-roberta-base", *, device: str = "cpu") -> None:
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install bimer[inference] to extract text features") from exc
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.model.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, texts: Sequence[str], *, batch_size: int = 16) -> np.ndarray:
        outputs: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            tokens = {key: value.to(self.device) for key, value in tokens.items()}
            hidden = self.model(**tokens).last_hidden_state
            pooled = mean_pool_hidden(hidden, tokens["attention_mask"])
            outputs.append(pooled.cpu().numpy().astype(np.float32))
        return np.concatenate(outputs) if outputs else np.empty((0, self.output_dim), np.float32)


class AudioFeatureExtractor:
    output_dim = 1024

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-xls-r-300m",
        *,
        device: str = "cpu",
    ) -> None:
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("Install bimer[inference] to extract audio features") from exc
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.model.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, waveforms: Sequence[np.ndarray]) -> np.ndarray:
        if not waveforms:
            return np.empty((0, self.output_dim), np.float32)
        inputs = self.processor(
            [np.asarray(waveform, np.float32) for waveform in waveforms],
            sampling_rate=16000,
            padding=True,
            return_tensors="pt",
        )
        input_values = inputs.input_values.to(self.device)
        attention_mask = getattr(inputs, "attention_mask", None)
        model_inputs: dict[str, Tensor] = {"input_values": input_values}
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask.to(self.device)
        hidden = self.model(**model_inputs).last_hidden_state
        if attention_mask is None:
            pooled = hidden.mean(dim=1)
        else:
            feature_mask = self.model._get_feature_vector_attention_mask(
                hidden.shape[1], model_inputs["attention_mask"]
            )
            pooled = mean_pool_hidden(hidden, feature_mask)
        return pooled.cpu().numpy().astype(np.float32)


def _probe_duration(video_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def read_uniform_video_frames(video_path: Path | str, *, count: int = 16) -> np.ndarray:
    path = Path(video_path)
    duration = _probe_duration(path)
    timestamps = np.linspace(0.0, max(0.0, duration - 0.001), count)
    frames: list[np.ndarray] = []
    for timestamp in timestamps:
        result = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-ss",
                f"{timestamp:.6f}",
                "-i",
                str(path),
                "-frames:v",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "png",
                "-",
            ],
            check=True,
            capture_output=True,
        )
        with Image.open(io.BytesIO(result.stdout)) as image:
            frames.append(np.asarray(image.convert("RGB")))
    return np.stack(frames)


def _center_square(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    size = min(height, width)
    top = (height - size) // 2
    left = (width - size) // 2
    return frame[top : top + size, left : left + size]


class YuNetFaceCropper:
    def __init__(self, model_path: Path | str, *, score_threshold: float = 0.8) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Install opencv-python-headless to use YuNet") from exc
        self.cv2 = cv2
        self.detector = cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            (320, 320),
            score_threshold,
            0.3,
            5000,
        )

    def crop_largest(self, frame: np.ndarray) -> tuple[np.ndarray, bool]:
        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(frame[:, :, ::-1])
        if faces is None or len(faces) == 0:
            return _center_square(frame), False
        face = max(faces, key=lambda item: float(item[2] * item[3]))
        x, y, w, h = face[:4].astype(int)
        margin = int(max(w, h) * 0.15)
        left, top = max(0, x - margin), max(0, y - margin)
        right, bottom = min(width, x + w + margin), min(height, y + h + margin)
        return frame[top:bottom, left:right], True


class VisionFeatureExtractor:
    output_dim = 512

    def __init__(self, *, device: str = "cpu") -> None:
        from torchvision.models.video import R3D_18_Weights, r3d_18

        self.device = torch.device(device)
        self.model = r3d_18(weights=R3D_18_Weights.DEFAULT)
        self.model.fc = nn.Identity()
        self.model = self.model.to(self.device).eval()
        self.model.requires_grad_(False)

    @torch.inference_mode()
    def encode_frames(self, frames: np.ndarray) -> np.ndarray:
        if frames.shape[0] != 16:
            raise ValueError("R3D-18 input must contain exactly 16 frames")
        tensor = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float() / 255.0
        tensor = torch.nn.functional.interpolate(
            tensor, size=(112, 112), mode="bilinear", align_corners=False
        )
        mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1)
        std = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        tensor = tensor.permute(1, 0, 2, 3).unsqueeze(0).to(self.device)
        return self.model(tensor).cpu().numpy().astype(np.float32)

    def encode_video(
        self,
        video_path: Path | str,
        *,
        face_cropper: YuNetFaceCropper,
        frame_drop_fraction: float = 0.0,
        seed: int = 42,
    ) -> tuple[np.ndarray, bool]:
        frames = read_uniform_video_frames(video_path, count=16)
        if frame_drop_fraction:
            frames = drop_video_frames(frames, fraction=frame_drop_fraction, seed=seed)
        crops: list[np.ndarray] = []
        detected: list[bool] = []
        for frame in frames:
            crop, found = face_cropper.crop_largest(frame)
            crops.append(crop)
            detected.append(found)
        available = vision_modality_available(detected)
        if not available:
            return np.zeros((1, self.output_dim), dtype=np.float32), False
        resized = []
        for crop in crops:
            image = Image.fromarray(crop).resize((112, 112))
            resized.append(np.asarray(image))
        return self.encode_frames(np.stack(resized)), True
