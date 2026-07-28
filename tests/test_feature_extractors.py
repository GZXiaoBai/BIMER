import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from bimer.feature_extractors import (
    AudioFeatureExtractor,
    VisionFeatureExtractor,
    YuNetFaceCropper,
    _prepare_clip_tensor,
    mean_pool_hidden,
    prepare_audio_waveforms,
    prepare_video_clip,
    prepare_video_clip_with_quality,
    prepare_video_segment_with_quality,
    read_uniform_video_frames,
    read_uniform_video_segment_frames,
    uniform_frame_indices,
    vision_modality_available,
)


def test_mean_pool_hidden_ignores_padding_tokens():
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    pooled = mean_pool_hidden(hidden, mask)
    assert torch.allclose(pooled, torch.tensor([[2.0, 2.0]]))


def test_uniform_frame_indices_return_exact_requested_count():
    assert uniform_frame_indices(5, 4).tolist() == [0, 1, 3, 4]
    repeated = uniform_frame_indices(2, 4)
    assert repeated.shape == (4,)
    assert repeated[0] == 0 and repeated[-1] == 1


def test_visual_modality_requires_four_detected_faces():
    assert vision_modality_available(np.array([1, 1, 1, 1], dtype=bool)) is True
    assert vision_modality_available(np.array([1, 1, 1, 0], dtype=bool)) is False


def test_audio_extractor_limits_each_inference_batch():
    class RecordingAudioExtractor(AudioFeatureExtractor):
        def __init__(self):
            self.batch_lengths = []

        def _encode_batch(self, waveforms):
            self.batch_lengths.append(len(waveforms))
            return np.zeros((len(waveforms), self.output_dim), dtype=np.float32)

    extractor = RecordingAudioExtractor()
    features = extractor.encode(
        [np.zeros(160, dtype=np.float32) for _ in range(5)],
        batch_size=2,
    )

    assert extractor.batch_lengths == [2, 2, 1]
    assert features.shape == (5, 1024)


def test_audio_extractor_pads_sub_receptive_field_waveforms():
    class RecordingProcessor:
        def __init__(self):
            self.lengths = []

        def __call__(self, waveforms, **_kwargs):
            self.lengths = [len(waveform) for waveform in waveforms]
            width = max(self.lengths)
            return SimpleNamespace(
                input_values=torch.zeros((len(waveforms), width)),
                attention_mask=None,
            )

    class RecordingModel:
        def __call__(self, **model_inputs):
            rows = model_inputs["input_values"].shape[0]
            return SimpleNamespace(last_hidden_state=torch.ones((rows, 1, 1024)))

    extractor = AudioFeatureExtractor.__new__(AudioFeatureExtractor)
    extractor.device = torch.device("cpu")
    extractor.processor = RecordingProcessor()
    extractor.model = RecordingModel()

    features = extractor._encode_batch(
        [
            np.ones(1, dtype=np.float32),
            np.ones(400, dtype=np.float32),
        ]
    )

    assert extractor.processor.lengths == [400, 400]
    assert features.shape == (2, 1024)


def test_audio_extractor_uses_feature_extractor_without_tokenizer(monkeypatch):
    feature_extractor = object()

    class FakeAutoFeatureExtractor:
        @classmethod
        def from_pretrained(cls, _model_name):
            return feature_extractor

    class FakeAutoProcessor:
        @classmethod
        def from_pretrained(cls, _model_name):
            raise AssertionError("XLS-R feature extraction must not load a tokenizer")

    class FakeModel:
        def to(self, _device):
            return self

        def eval(self):
            return self

        def requires_grad_(self, _requires_grad):
            return self

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, _model_name):
            return FakeModel()

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoFeatureExtractor = FakeAutoFeatureExtractor
    fake_transformers.AutoModel = FakeAutoModel
    fake_transformers.AutoProcessor = FakeAutoProcessor
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    extractor = AudioFeatureExtractor()

    assert extractor.processor is feature_extractor


def test_vision_extractor_loads_explicit_offline_weights(tmp_path, monkeypatch):
    recorded = {}

    class FakeModel:
        def __init__(self):
            self.fc = object()

        def load_state_dict(self, state):
            recorded["state"] = state

        def to(self, device):
            recorded["device"] = str(device)
            return self

        def eval(self):
            return self

        def requires_grad_(self, value):
            recorded["requires_grad"] = value
            return self

    def fake_r3d_18(*, weights):
        recorded["weights"] = weights
        return FakeModel()

    fake_video = ModuleType("torchvision.models.video")
    fake_video.R3D_18_Weights = SimpleNamespace(DEFAULT=object())
    fake_video.r3d_18 = fake_r3d_18
    monkeypatch.setitem(sys.modules, "torchvision.models.video", fake_video)
    monkeypatch.setattr(
        torch,
        "load",
        lambda path, **_kwargs: {
            "offline": Path(path).name,
        },
    )
    weights = tmp_path / "r3d.pt"
    weights.write_bytes(b"weights")

    VisionFeatureExtractor(device="cpu", weights_path=weights)

    assert recorded["weights"] is None
    assert recorded["state"] == {"offline": "r3d.pt"}
    assert recorded["requires_grad"] is False


def test_video_reader_uses_one_capture_and_uniform_frame_positions(monkeypatch):
    frames = [np.full((2, 3, 3), [index, 0, 0], dtype=np.uint8) for index in range(5)]

    class FakeCapture:
        def __init__(self):
            self.position = 0
            self.requested_positions = []
            self.released = False

        def isOpened(self):
            return True

        def get(self, _property):
            return len(frames)

        def set(self, _property, value):
            self.position = int(value)
            self.requested_positions.append(self.position)

        def read(self):
            return True, frames[self.position]

        def release(self):
            self.released = True

    capture = FakeCapture()
    fake_cv2 = SimpleNamespace(
        CAP_PROP_FRAME_COUNT=7,
        CAP_PROP_POS_FRAMES=1,
        VideoCapture=lambda _path: capture,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    decoded = read_uniform_video_frames("video.mp4", count=4)

    assert capture.requested_positions == [0, 1, 3, 4]
    assert capture.released is True
    assert decoded.shape == (4, 2, 3, 3)
    assert decoded[-1, 0, 0].tolist() == [0, 0, 4]


def test_segment_video_reader_uses_timestamp_bounded_positions(monkeypatch):
    frames = [np.full((2, 2, 3), index, dtype=np.uint8) for index in range(100)]

    class FakeCapture:
        def __init__(self):
            self.position = 0
            self.requested_positions = []

        def isOpened(self):
            return True

        def get(self, prop):
            return 100 if prop == 7 else 10

        def set(self, _prop, value):
            self.position = int(value)
            self.requested_positions.append(self.position)

        def read(self):
            return True, frames[self.position]

        def release(self):
            pass

    capture = FakeCapture()
    fake_cv2 = SimpleNamespace(
        CAP_PROP_FRAME_COUNT=7,
        CAP_PROP_FPS=5,
        CAP_PROP_POS_FRAMES=1,
        VideoCapture=lambda _path: capture,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    decoded = read_uniform_video_segment_frames(
        "video.mp4", start_seconds=2.0, end_seconds=4.0, count=4
    )

    assert capture.requested_positions == [20, 26, 33, 39]
    assert decoded.shape == (4, 2, 2, 3)


def test_vision_extractor_batches_clips_in_order():
    class RecordingVision(VisionFeatureExtractor):
        def __init__(self):
            self.batch_lengths = []

        def _encode_clip_batch(self, clips):
            self.batch_lengths.append(len(clips))
            return np.asarray(
                [[clip[0, 0, 0, 0]] * self.output_dim for clip in clips],
                dtype=np.float32,
            )

    clips = [np.full((16, 112, 112, 3), value, dtype=np.uint8) for value in range(5)]
    extractor = RecordingVision()
    result = extractor.encode_clips(clips, batch_size=2)

    assert extractor.batch_lengths == [2, 2, 1]
    assert result[:, 0].tolist() == [0, 1, 2, 3, 4]


def test_prepare_video_clip_marks_three_faces_unavailable(monkeypatch):
    frames = np.ones((16, 8, 8, 3), dtype=np.uint8)
    monkeypatch.setattr(
        "bimer.feature_extractors.read_uniform_video_frames",
        lambda *_args, **_kwargs: frames,
    )

    class ThreeFaceCropper:
        def __init__(self):
            self.calls = 0

        def crop_largest(self, frame):
            self.calls += 1
            return frame, self.calls <= 3

    clip, available = prepare_video_clip(
        "video.mp4",
        face_cropper=ThreeFaceCropper(),
    )

    assert clip.shape == (16, 112, 112, 3)
    assert available is False


def test_prepare_video_clip_returns_continuous_visual_quality(monkeypatch):
    frames = np.ones((16, 8, 8, 3), dtype=np.uint8)
    monkeypatch.setattr(
        "bimer.feature_extractors.read_uniform_video_frames",
        lambda *_args, **_kwargs: frames,
    )

    class StableFaceCropper:
        def crop_largest_with_metadata(self, frame):
            return frame, True, (0.25, 0.25, 0.5, 0.5)

    clip, available, quality = prepare_video_clip_with_quality(
        "video.mp4", face_cropper=StableFaceCropper()
    )

    assert clip.shape == (16, 112, 112, 3)
    assert available is True
    np.testing.assert_allclose(quality, [1.0, 1.0, 1.0, 0.25])


def test_feature_extractors_validate_empty_and_invalid_batches():
    with pytest.raises(ValueError, match="minimum_samples"):
        prepare_audio_waveforms([np.ones(2)], minimum_samples=0)
    prepared, available = prepare_audio_waveforms(
        [np.ones(399), np.ones(400)],
    )
    assert [len(item) for item in prepared] == [400, 400]
    assert available.tolist() == [False, True]
    with pytest.raises(ValueError, match="positive"):
        uniform_frame_indices(0)

    assert AudioFeatureExtractor.__new__(AudioFeatureExtractor).encode([]).shape == (
        0,
        1024,
    )
    with pytest.raises(ValueError, match="batch_size"):
        AudioFeatureExtractor.__new__(AudioFeatureExtractor).encode(
            [np.ones(400)],
            batch_size=0,
        )
    with pytest.raises(ValueError, match="16 RGB"):
        _prepare_clip_tensor([np.zeros((2, 2, 2, 3), dtype=np.uint8)])
    with pytest.raises(ValueError, match="batch_size"):
        VisionFeatureExtractor.__new__(VisionFeatureExtractor).encode_clips(
            [],
            batch_size=0,
        )


def test_video_readers_report_open_metadata_and_frame_failures(monkeypatch):
    class ClosedCapture:
        def isOpened(self):
            return False

        def release(self):
            self.released = True

    fake_cv2 = SimpleNamespace(
        CAP_PROP_FRAME_COUNT=7,
        CAP_PROP_FPS=5,
        CAP_PROP_POS_FRAMES=1,
        VideoCapture=lambda _path: ClosedCapture(),
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    with pytest.raises(RuntimeError, match="Cannot open"):
        read_uniform_video_frames("closed.mp4")
    with pytest.raises(RuntimeError, match="Cannot open"):
        read_uniform_video_segment_frames(
            "closed.mp4",
            start_seconds=0,
            end_seconds=1,
        )
    with pytest.raises(ValueError, match="timestamps"):
        read_uniform_video_segment_frames(
            "closed.mp4",
            start_seconds=1,
            end_seconds=1,
        )

    class BadCapture:
        def __init__(self, *, frames: int, fps: int, readable: bool = True):
            self.frames = frames
            self.fps = fps
            self.readable = readable

        def isOpened(self):
            return True

        def get(self, prop):
            return self.frames if prop == 7 else self.fps

        def set(self, _prop, _value):
            pass

        def read(self):
            return self.readable, None

        def release(self):
            pass

    fake_cv2.VideoCapture = lambda _path: BadCapture(frames=0, fps=0)
    with pytest.raises(RuntimeError, match="no readable frames"):
        read_uniform_video_frames("empty.mp4")
    with pytest.raises(RuntimeError, match="invalid frame metadata"):
        read_uniform_video_segment_frames(
            "bad-meta.mp4",
            start_seconds=0,
            end_seconds=1,
        )

    fake_cv2.VideoCapture = lambda _path: BadCapture(frames=2, fps=1, readable=False)
    with pytest.raises(RuntimeError, match="Cannot read frame"):
        read_uniform_video_frames("broken.mp4", count=1)
    with pytest.raises(RuntimeError, match="Cannot read frame"):
        read_uniform_video_segment_frames(
            "broken.mp4",
            start_seconds=0,
            end_seconds=1,
            count=1,
        )


def test_yunet_cropper_handles_no_face_and_largest_face(monkeypatch):
    class Detector:
        def setInputSize(self, size):
            self.size = size

        def detect(self, _frame):
            return None, self.faces

    detector = Detector()
    fake_cv2 = SimpleNamespace(
        FaceDetectorYN=SimpleNamespace(create=lambda *_args: detector),
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    cropper = YuNetFaceCropper("yunet.onnx")
    frame = np.zeros((8, 12, 3), dtype=np.uint8)

    detector.faces = None
    crop, found, bbox = cropper.crop_largest_with_metadata(frame)
    assert crop.shape == (8, 8, 3)
    assert found is False and bbox is None

    detector.faces = np.asarray(
        [[1, 1, 2, 2, 0.9], [2, 1, 6, 4, 0.8]],
        dtype=np.float32,
    )
    crop, found = cropper.crop_largest(frame)
    assert crop.size > 0
    assert found is True


def test_segment_preparation_and_video_wrappers_cover_quality_paths(monkeypatch):
    frames = np.ones((16, 8, 8, 3), dtype=np.uint8)
    monkeypatch.setattr(
        "bimer.feature_extractors.read_uniform_video_segment_frames",
        lambda *_args, **_kwargs: frames,
    )

    class Cropper:
        def crop_largest_with_metadata(self, frame):
            return frame, True, (0.1, 0.1, 0.5, 0.5)

    clip, available, quality = prepare_video_segment_with_quality(
        "video.mp4",
        start_seconds=0,
        end_seconds=1,
        face_cropper=Cropper(),
        frame_drop_fraction=0.25,
    )
    assert clip.shape == (16, 112, 112, 3)
    assert available is True
    assert quality.shape == (4,)

    extractor = VisionFeatureExtractor.__new__(VisionFeatureExtractor)
    extractor.encode_video_with_quality = lambda *_args, **_kwargs: (
        np.ones((1, 512), dtype=np.float32),
        True,
        quality,
    )
    feature, available = extractor.encode_video(
        "video.mp4",
        face_cropper=Cropper(),
    )
    assert feature.shape == (1, 512)
    assert available is True
