import os
import time

import numpy as np

from bimer.runtime_cache import RuntimeFeatureCache


def test_cache_hit_and_text_edit_only_invalidates_text_namespace(tmp_path):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"video")
    cache = RuntimeFeatureCache(tmp_path / "cache", max_bytes=1024 * 1024, ttl_seconds=3600)
    timestamps = [(0.0, 2.0)]
    media_payload = {
        "video_sha256": cache.file_sha256(video),
        "timestamps": timestamps,
    }
    audio_key = cache.key("audio", {**media_payload, "encoder": "xlsr"})
    old_text_key = cache.key(
        "text",
        {**media_payload, "texts": ["old"], "encoder": "xlmr"},
    )
    new_text_key = cache.key(
        "text",
        {**media_payload, "texts": ["new"], "encoder": "xlmr"},
    )
    cache.store(audio_key, {"features": np.ones((1, 2), np.float32)})
    cache.store(old_text_key, {"features": np.ones((1, 3), np.float32)})

    assert cache.load(audio_key) is not None
    assert cache.load(old_text_key) is not None
    assert cache.load(new_text_key) is None


def test_timestamp_change_invalidates_media_features_and_writes_atomically(tmp_path):
    cache = RuntimeFeatureCache(tmp_path / "cache", max_bytes=1024 * 1024, ttl_seconds=3600)
    old_key = cache.key("vision", {"sha": "x", "timestamps": [(0.0, 1.0)]})
    new_key = cache.key("vision", {"sha": "x", "timestamps": [(0.0, 2.0)]})

    cache.store(old_key, {"features": np.ones((2, 2), np.float32)})

    assert cache.load(old_key)["features"].shape == (2, 2)
    assert cache.load(new_key) is None
    assert not list((tmp_path / "cache").glob("*.tmp"))


def test_cache_expires_and_clear_removes_entries(tmp_path):
    cache = RuntimeFeatureCache(tmp_path / "cache", max_bytes=1024 * 1024, ttl_seconds=0.01)
    key = cache.key("text", {"value": 1})
    path = cache.store(key, {"features": np.ones((1, 1), np.float32)})
    old = time.time() - 5
    os.utime(path, (old, old))

    assert cache.load(key) is None
    cache.store(key, {"features": np.ones((1, 1), np.float32)})
    assert cache.clear() == 1
    assert cache.load(key) is None
