import numpy as np

from bimer.calibration import (
    apply_temperature,
    calibration_metrics,
    fit_calibration_profile,
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
