from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn

from .robustness import drop_video_frames
from .quality import vision_quality


WAV2VEC2_MIN_INPUT_SAMPLES = 400


def prepare_audio_waveforms(
    waveforms: Sequence[np.ndarray],
    *,
    minimum_samples: int = WAV2VEC2_MIN_INPUT_SAMPLES,
) -> tuple[list[np.ndarray], np.ndarray]:
    if minimum_samples <= 0:
        raise ValueError("minimum_samples must be positive")
    prepared: list[np.ndarray] = []
    available: list[bool] = []
    for waveform in waveforms:
        signal = np.asarray(waveform, dtype=np.float32).reshape(-1)
        is_available = signal.size >= minimum_samples
        prepared.append(
            signal if is_available else np.zeros(minimum_samples, dtype=np.float32)
        )
        available.append(is_available)
    return prepared, np.asarray(available, dtype=np.bool_)


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
            from transformers import AutoFeatureExtractor, AutoModel
        except ImportError as exc:
            raise RuntimeError("Install bimer[inference] to extract audio features") from exc
        self.device = torch.device(device)
        self.processor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.model.requires_grad_(False)

    def _encode_batch(self, waveforms: Sequence[np.ndarray]) -> np.ndarray:
        prepared, _ = prepare_audio_waveforms(waveforms)
        inputs = self.processor(
            prepared,
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

    @torch.inference_mode()
    def encode(
        self,
        waveforms: Sequence[np.ndarray],
        *,
        batch_size: int = 4,
    ) -> np.ndarray:
        if not waveforms:
            return np.empty((0, self.output_dim), np.float32)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        outputs = [
            self._encode_batch(waveforms[start : start + batch_size])
            for start in range(0, len(waveforms), batch_size)
        ]
        return np.concatenate(outputs)


def read_uniform_video_frames(video_path: Path | str, *, count: int = 16) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Install opencv-python-headless to read video frames") from exc

    path = Path(video_path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open video {path}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        capture.release()
        raise RuntimeError(f"Video has no readable frames: {path}")
    frames: list[np.ndarray] = []
    try:
        for index in uniform_frame_indices(total_frames, count):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            readable, frame = capture.read()
            if not readable or frame is None:
                raise RuntimeError(f"Cannot read frame {index} from {path}")
            frames.append(frame[:, :, ::-1].copy())
    finally:
        capture.release()
    return np.stack(frames)


def read_uniform_video_segment_frames(
    video_path: Path | str,
    *,
    start_seconds: float,
    end_seconds: float,
    count: int = 16,
) -> np.ndarray:
    if start_seconds < 0 or end_seconds <= start_seconds:
        raise ValueError("video segment timestamps are invalid")
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Install opencv-python-headless to read video frames") from exc
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open video {video_path}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if total_frames <= 0 or fps <= 0:
        capture.release()
        raise RuntimeError(f"Video has invalid frame metadata: {video_path}")
    first = min(total_frames - 1, max(0, int(round(start_seconds * fps))))
    last = min(total_frames - 1, max(first, int(round(end_seconds * fps)) - 1))
    positions = np.rint(np.linspace(first, last, count)).astype(np.int64)
    frames: list[np.ndarray] = []
    try:
        for position in positions:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
            readable, frame = capture.read()
            if not readable or frame is None:
                raise RuntimeError(f"Cannot read frame {position} from {video_path}")
            frames.append(frame[:, :, ::-1].copy())
    finally:
        capture.release()
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
        crop, found, _ = self.crop_largest_with_metadata(frame)
        return crop, found

    def crop_largest_with_metadata(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, bool, tuple[float, float, float, float] | None]:
        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(frame[:, :, ::-1])
        if faces is None or len(faces) == 0:
            return _center_square(frame), False, None
        face = max(faces, key=lambda item: float(item[2] * item[3]))
        x, y, w, h = face[:4].astype(int)
        margin = int(max(w, h) * 0.15)
        left, top = max(0, x - margin), max(0, y - margin)
        right, bottom = min(width, x + w + margin), min(height, y + h + margin)
        normalized_bbox = (
            float(x / max(1, width)),
            float(y / max(1, height)),
            float(w / max(1, width)),
            float(h / max(1, height)),
        )
        return frame[top:bottom, left:right], True, normalized_bbox


def prepare_video_clip(
    video_path: Path | str,
    *,
    face_cropper: YuNetFaceCropper,
    frame_drop_fraction: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, bool]:
    clip, available, _ = prepare_video_clip_with_quality(
        video_path,
        face_cropper=face_cropper,
        frame_drop_fraction=frame_drop_fraction,
        seed=seed,
    )
    return clip, available


def prepare_video_clip_with_quality(
    video_path: Path | str,
    *,
    face_cropper: YuNetFaceCropper,
    frame_drop_fraction: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, bool, np.ndarray]:
    frames = read_uniform_video_frames(video_path, count=16)
    return _prepare_sampled_frames(
        frames,
        face_cropper=face_cropper,
        frame_drop_fraction=frame_drop_fraction,
        seed=seed,
    )


def prepare_video_segment_with_quality(
    video_path: Path | str,
    *,
    start_seconds: float,
    end_seconds: float,
    face_cropper: YuNetFaceCropper,
    frame_drop_fraction: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, bool, np.ndarray]:
    frames = read_uniform_video_segment_frames(
        video_path,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        count=16,
    )
    return _prepare_sampled_frames(
        frames,
        face_cropper=face_cropper,
        frame_drop_fraction=frame_drop_fraction,
        seed=seed,
    )


def _prepare_sampled_frames(
    frames: np.ndarray,
    *,
    face_cropper: YuNetFaceCropper,
    frame_drop_fraction: float,
    seed: int,
) -> tuple[np.ndarray, bool, np.ndarray]:
    if frame_drop_fraction:
        frames = drop_video_frames(frames, fraction=frame_drop_fraction, seed=seed)
    prepared: list[np.ndarray] = []
    detected: list[bool] = []
    bboxes: list[tuple[float, float, float, float] | None] = []
    for frame in frames:
        metadata_cropper = getattr(face_cropper, "crop_largest_with_metadata", None)
        if metadata_cropper is None:
            crop, found = face_cropper.crop_largest(frame)
            bbox = None
        else:
            crop, found, bbox = metadata_cropper(frame)
        prepared.append(np.asarray(Image.fromarray(crop).resize((112, 112))))
        detected.append(found)
        bboxes.append(bbox)
    quality = vision_quality(detected, bboxes, expected_frames=16)
    return np.stack(prepared), vision_modality_available(detected), quality


def _prepare_clip_tensor(clips: Sequence[np.ndarray]) -> Tensor:
    if any(
        np.asarray(clip).ndim != 4
        or np.asarray(clip).shape[0] != 16
        or np.asarray(clip).shape[-1] != 3
        for clip in clips
    ):
        raise ValueError("R3D-18 inputs must contain 16 RGB frames")
    array = np.stack(clips)
    tensor = torch.from_numpy(array.copy()).permute(0, 1, 4, 2, 3).float() / 255.0
    batch, frames, channels, height, width = tensor.shape
    tensor = torch.nn.functional.interpolate(
        tensor.reshape(batch * frames, channels, height, width),
        size=(112, 112),
        mode="bilinear",
        align_corners=False,
    ).reshape(batch, frames, channels, 112, 112)
    mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 1, 3, 1, 1)
    return ((tensor - mean) / std).permute(0, 2, 1, 3, 4)


class VisionFeatureExtractor:
    output_dim = 512

    def __init__(
        self,
        *,
        device: str = "cpu",
        weights_path: Path | str | None = None,
    ) -> None:
        from torchvision.models.video import R3D_18_Weights, r3d_18

        self.device = torch.device(device)
        if weights_path is None:
            self.model = r3d_18(weights=R3D_18_Weights.DEFAULT)
        else:
            self.model = r3d_18(weights=None)
            self.model.load_state_dict(
                torch.load(
                    weights_path,
                    map_location="cpu",
                    weights_only=True,
                )
            )
        self.model.fc = nn.Identity()
        self.model = self.model.to(self.device).eval()
        self.model.requires_grad_(False)

    @torch.inference_mode()
    def _encode_clip_batch(self, clips: Sequence[np.ndarray]) -> np.ndarray:
        tensor = _prepare_clip_tensor(clips).to(self.device)
        return self.model(tensor).cpu().numpy().astype(np.float32)

    def encode_clips(
        self,
        clips: Sequence[np.ndarray],
        *,
        batch_size: int = 8,
    ) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        outputs = [
            self._encode_clip_batch(clips[start : start + batch_size])
            for start in range(0, len(clips), batch_size)
        ]
        return (
            np.concatenate(outputs)
            if outputs
            else np.empty((0, self.output_dim), dtype=np.float32)
        )

    def encode_frames(self, frames: np.ndarray) -> np.ndarray:
        return self.encode_clips([frames], batch_size=1)

    def encode_video(
        self,
        video_path: Path | str,
        *,
        face_cropper: YuNetFaceCropper,
        frame_drop_fraction: float = 0.0,
        seed: int = 42,
    ) -> tuple[np.ndarray, bool]:
        feature, available, _ = self.encode_video_with_quality(
            video_path,
            face_cropper=face_cropper,
            frame_drop_fraction=frame_drop_fraction,
            seed=seed,
        )
        return feature, available

    def encode_video_with_quality(
        self,
        video_path: Path | str,
        *,
        face_cropper: YuNetFaceCropper,
        frame_drop_fraction: float = 0.0,
        seed: int = 42,
    ) -> tuple[np.ndarray, bool, np.ndarray]:
        clip, available, quality = prepare_video_clip_with_quality(
            video_path,
            face_cropper=face_cropper,
            frame_drop_fraction=frame_drop_fraction,
            seed=seed,
        )
        if not available:
            return np.zeros((1, self.output_dim), dtype=np.float32), False, quality
        return self.encode_frames(clip), True, quality

    def encode_video_segment_with_quality(
        self,
        video_path: Path | str,
        *,
        start_seconds: float,
        end_seconds: float,
        face_cropper: YuNetFaceCropper,
        frame_drop_fraction: float = 0.0,
        seed: int = 42,
    ) -> tuple[np.ndarray, bool, np.ndarray]:
        clip, available, quality = prepare_video_segment_with_quality(
            video_path,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            face_cropper=face_cropper,
            frame_drop_fraction=frame_drop_fraction,
            seed=seed,
        )
        if not available:
            return np.zeros((1, self.output_dim), dtype=np.float32), False, quality
        return self.encode_frames(clip), True, quality
