import numpy as np

from bimer.robustness import add_noise_at_snr, drop_video_frames, mask_feature_modality


def test_audio_noise_matches_requested_snr_with_seed():
    waveform = np.sin(np.linspace(0, 4 * np.pi, 1600)).astype(np.float32)
    noisy = add_noise_at_snr(waveform, snr_db=20.0, seed=42)
    signal_power = np.mean(waveform**2)
    noise_power = np.mean((noisy - waveform) ** 2)
    measured = 10 * np.log10(signal_power / noise_power)
    np.testing.assert_allclose(measured, 20.0, atol=0.25)


def test_frame_drop_zeros_exact_fraction_deterministically():
    frames = np.ones((16, 2, 2, 3), dtype=np.uint8)
    dropped = drop_video_frames(frames, fraction=0.25, seed=7)
    assert np.sum(np.all(dropped == 0, axis=(1, 2, 3))) == 4
    assert np.array_equal(dropped, drop_video_frames(frames, fraction=0.25, seed=7))


def test_mask_feature_modality_clears_values_and_mask():
    feature = np.ones((2, 5), dtype=np.float32)
    mask = np.ones((2, 3), dtype=np.bool_)
    cleared, updated = mask_feature_modality(feature, mask, modality_index=2)
    assert np.all(cleared == 0)
    assert updated[:, 2].tolist() == [False, False]
