"""Tests for psychometric fitting and choice-coding resolution.

Covers cumulative_gaussian, fit_psychometric, choice_proportions,
PsychometricFit.predict, and resolve_choice_coding.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from seven_a.behavior import (
    ChoiceCoding,
    PsychometricFit,
    choice_proportions,
    cumulative_gaussian,
    fit_psychometric,
    resolve_choice_coding,
)


# ------------------------------------------------------------------
# cumulative_gaussian
# ------------------------------------------------------------------
class TestCumulativeGaussian:
    """Properties of the parametric psychometric function."""

    def test_value_at_mu(self):
        """P(right) at the PSE should be 0.5 (no lapse)."""
        mu, sigma = 2.0, 5.0
        assert cumulative_gaussian(np.array([mu]), mu, sigma, lapse=0.0) == pytest.approx(
            0.5, abs=1e-9
        )

    def test_monotonically_increasing(self):
        """The psychometric curve must be non-decreasing."""
        x = np.linspace(-20, 20, 200)
        p = cumulative_gaussian(x, mu=0.0, sigma=5.0, lapse=0.0)
        assert np.all(np.diff(p) >= -1e-12)

    def test_lapse_bounds_output(self):
        """With lapse > 0 the output should stay within [lapse/2, 1 - lapse/2]."""
        lapse = 0.1
        x = np.linspace(-100, 100, 500)
        p = cumulative_gaussian(x, mu=0.0, sigma=5.0, lapse=lapse)
        assert p.min() >= lapse / 2 - 1e-9
        assert p.max() <= 1.0 - lapse / 2 + 1e-9

    def test_symmetry_about_mu(self):
        """P(mu - d) + P(mu + d) should equal 1 when lapse is 0."""
        mu, sigma = 3.0, 4.0
        d = np.array([1.0, 5.0, 10.0])
        left = cumulative_gaussian(mu - d, mu, sigma, lapse=0.0)
        right = cumulative_gaussian(mu + d, mu, sigma, lapse=0.0)
        np.testing.assert_allclose(left + right, 1.0, atol=1e-9)


# ------------------------------------------------------------------
# fit_psychometric
# ------------------------------------------------------------------
class TestFitPsychometric:
    """Recovery of known parameters from synthetic data."""

    @pytest.fixture()
    def synthetic_sigmoid(self):
        """Generate binary choices from a known cumulative Gaussian."""
        rng = np.random.default_rng(42)
        true_mu, true_sigma = 1.0, 6.0
        headings = np.repeat(np.linspace(-20, 20, 9), 80)  # 720 trials
        p_right = cumulative_gaussian(headings, true_mu, true_sigma, lapse=0.0)
        choices = (rng.random(len(headings)) < p_right).astype(float)
        return headings, choices, true_mu, true_sigma

    def test_recovers_known_parameters(self, synthetic_sigmoid):
        """Fitted mu and sigma should be close to the generative values."""
        headings, choices, true_mu, true_sigma = synthetic_sigmoid
        fit = fit_psychometric(headings, choices, fit_lapse=True)
        assert fit.converged
        assert fit.mu == pytest.approx(true_mu, abs=2.0)
        assert fit.sigma == pytest.approx(true_sigma, abs=3.0)

    def test_converged_flag(self, synthetic_sigmoid):
        """The optimiser should report convergence on well-behaved data."""
        headings, choices, _, _ = synthetic_sigmoid
        fit = fit_psychometric(headings, choices)
        assert fit.converged is True

    def test_too_few_trials_returns_nan(self):
        """Fewer than 4 trials should yield NaN parameters and converged=False."""
        fit = fit_psychometric([0.0, 1.0], [0.0, 1.0])
        assert not fit.converged
        assert np.isnan(fit.mu)
        assert np.isnan(fit.sigma)

    def test_single_choice_value_returns_nan(self):
        """All-same choices (no variance) should fail gracefully."""
        headings = np.linspace(-10, 10, 20)
        choices = np.ones(20)
        fit = fit_psychometric(headings, choices)
        assert not fit.converged
        assert np.isnan(fit.mu)

    def test_bootstrap_ci(self, synthetic_sigmoid):
        """Bootstrap should produce mu and sigma intervals that contain the true values."""
        headings, choices, true_mu, true_sigma = synthetic_sigmoid
        fit = fit_psychometric(headings, choices, fit_lapse=False, n_boot=200, rng=np.random.default_rng(7))
        assert "mu" in fit.ci
        assert "sigma" in fit.ci
        mu_lo, mu_hi = fit.ci["mu"]
        assert mu_lo < true_mu < mu_hi


# ------------------------------------------------------------------
# choice_proportions
# ------------------------------------------------------------------
class TestChoiceProportions:
    """Binning of per-heading rightward proportions."""

    def test_correct_proportions(self):
        """Check that proportions match hand-computed values."""
        headings = np.array([-10, -10, -10, 0, 0, 10, 10, 10, 10])
        rightward = np.array([0, 0, 1, 0, 1, 1, 1, 1, 0])
        levels, prop, n = choice_proportions(headings, rightward)
        np.testing.assert_array_equal(levels, [-10, 0, 10])
        np.testing.assert_allclose(prop, [1 / 3, 0.5, 0.75])
        np.testing.assert_array_equal(n, [3, 2, 4])


# ------------------------------------------------------------------
# PsychometricFit.predict
# ------------------------------------------------------------------
class TestPsychometricFitPredict:
    """The predict method should reproduce cumulative_gaussian."""

    def test_predict_matches_cumulative_gaussian(self):
        fit = PsychometricFit(mu=1.0, sigma=5.0, lapse=0.05, n_trials=100, loglik=-50.0, converged=True)
        x = np.linspace(-15, 15, 50)
        expected = cumulative_gaussian(x, 1.0, 5.0, 0.05)
        np.testing.assert_allclose(fit.predict(x), expected)


# ------------------------------------------------------------------
# resolve_choice_coding
# ------------------------------------------------------------------
class TestResolveChoiceCoding:
    """Detection of absolute vs. stimulus-relative coding."""

    @pytest.fixture()
    def absolute_data(self):
        """Synthetic data where the stored choice already means rightward.

        Uses a non-zero mu so the absolute and stimulus-relative mappings
        produce distinguishable curves (at mu=0 with symmetric headings
        they are equivalent).
        """
        rng = np.random.default_rng(99)
        headings = np.repeat(np.linspace(-20, 20, 9), 120)
        p_right = cumulative_gaussian(headings, mu=3.0, sigma=6.0)
        rightward = (rng.random(len(headings)) < p_right).astype(float)
        # Encode as -5 / 5 where 5 = rightward
        choice = np.where(rightward, 5.0, -5.0)
        return headings, choice

    def test_identifies_absolute_coding(self, absolute_data):
        """An already-sigmoidal mapping should be classified as absolute."""
        headings, choice = absolute_data
        coding = resolve_choice_coding(headings, choice)
        assert coding.scheme == "absolute"
        assert coding.values == (-5.0, 5.0)

    def test_identifies_stimulus_relative_coding(self):
        """A stimulus-relative (accuracy-like) coding should be recovered."""
        rng = np.random.default_rng(12)
        headings = np.repeat(np.linspace(-20, 20, 9), 100)
        p_right = cumulative_gaussian(headings, mu=0.0, sigma=6.0)
        rightward = (rng.random(len(headings)) < p_right).astype(float)
        # Encode as "correct" (1) / "error" (0) relative to the stimulus:
        # On positive headings, rightward is correct; on negative, leftward is correct.
        correct = np.where(headings > 0, rightward, 1.0 - rightward)
        # At heading == 0 define randomly (will be excluded by default)
        coding = resolve_choice_coding(headings, correct)
        assert coding.scheme == "stimulus_relative"

    def test_raises_on_three_choice_values(self):
        """More than two distinct choice values should raise ValueError."""
        headings = np.array([1.0, 2.0, 3.0])
        choice = np.array([0.0, 1.0, 2.0])
        with pytest.raises(ValueError, match="expected exactly 2"):
            resolve_choice_coding(headings, choice)
