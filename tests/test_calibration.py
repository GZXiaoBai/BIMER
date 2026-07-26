import numpy as np
import pytest

from bimer.calibration import (
    CalibrationProfile,
    apply_temperature,
    calibration_metrics,
    fit_calibration_profile,
    fit_temperature,
    select_uncertainty_threshold,
)


def test_temperature_scaling_preserves_argmax_and_metrics_are_finite():
    probabilities = np.asarray(
        [[0.90, 0.10], [0.80, 0.20], [0.25, 0.75], [0.05, 0.95]],
        dtype=np.float64,
    )
    truth = np.asarray([0, 1, 1, 1])

    calibrated = apply_temperature(probabilities, 2.0)
    metrics = calibration_metrics(calibrated, truth, bins=15)

    np.testing.assert_array_equal(
        probabilities.argmax(axis=1),
        calibrated.argmax(axis=1),
    )
    assert set(metrics) == {"ece", "brier", "nll", "accuracy"}
    assert all(np.isfinite(value) for value in metrics.values())


def test_language_profile_only_enables_temperature_when_acceptance_rule_passes():
    probabilities = np.asarray(
        [
            [0.99, 0.01],
            [0.99, 0.01],
            [0.01, 0.99],
            [0.01, 0.99],
            [0.60, 0.40],
            [0.40, 0.60],
            [0.55, 0.45],
            [0.45, 0.55],
        ]
    )
    truth = np.asarray([0, 1, 1, 0, 0, 1, 0, 1])
    languages = np.asarray(["en"] * 4 + ["zh"] * 4)

    profile = fit_calibration_profile(probabilities, truth, languages)

    assert set(profile.languages) == {"en", "zh"}
    for language in ("en", "zh"):
        result = profile.languages[language]
        if result.enabled:
            assert result.after["ece"] <= result.before["ece"] * 0.9
            assert result.after["nll"] <= result.before["nll"] + 1e-9
        else:
            assert result.temperature == 1.0


def test_uncertainty_threshold_requires_coverage_and_meaningful_accuracy_gain():
    probabilities = np.asarray(
        [
            [0.95, 0.05],
            [0.90, 0.10],
            [0.80, 0.20],
            [0.70, 0.30],
            [0.55, 0.45],
            [0.52, 0.48],
            [0.51, 0.49],
            [0.50, 0.50],
            [0.49, 0.51],
            [0.48, 0.52],
        ]
    )
    truth = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    threshold = select_uncertainty_threshold(probabilities, truth)
    confidence = probabilities.max(axis=1)

    assert 0.35 <= threshold <= 0.80
    assert float((confidence >= threshold).mean()) >= 0.70


@pytest.mark.parametrize(
    "probabilities",
    [
        np.asarray([0.5, 0.5]),
        np.asarray([[0.5]]),
        np.asarray([[np.nan, 0.5]]),
        np.asarray([[-0.1, 1.1]]),
        np.asarray([[0.0, 0.0]]),
    ],
)
def test_probability_validation_rejects_invalid_inputs(probabilities):
    with pytest.raises(ValueError):
        apply_temperature(probabilities, 1.0)


def test_temperature_and_metric_validation_rejects_invalid_arguments():
    probabilities = np.asarray([[0.7, 0.3], [0.2, 0.8]])
    with pytest.raises(ValueError, match="temperature"):
        apply_temperature(probabilities, 0.0)
    with pytest.raises(ValueError, match="truth"):
        calibration_metrics(probabilities, np.asarray([0]))
    with pytest.raises(ValueError, match="out-of-range"):
        calibration_metrics(probabilities, np.asarray([0, 2]))
    with pytest.raises(ValueError, match="bins"):
        calibration_metrics(probabilities, np.asarray([0, 1]), bins=0)
    with pytest.raises(ValueError, match="same length"):
        fit_temperature(probabilities, np.asarray([0]))


def test_calibration_profile_round_trip(tmp_path):
    probabilities = np.asarray([[0.90, 0.10], [0.60, 0.40], [0.20, 0.80], [0.45, 0.55]])
    truth = np.asarray([0, 1, 1, 0])
    profile = fit_calibration_profile(
        probabilities,
        truth,
        np.asarray(["zh", "zh", "en", "en"]),
    )

    path = profile.save(tmp_path / "nested" / "calibration.json")
    restored = CalibrationProfile.load(path)

    assert restored == profile
    assert restored.to_dict() == profile.to_dict()


def test_profile_rejects_misaligned_languages():
    with pytest.raises(ValueError, match="align"):
        fit_calibration_profile(
            np.asarray([[0.7, 0.3], [0.2, 0.8]]),
            np.asarray([0, 1]),
            np.asarray(["en"]),
        )
