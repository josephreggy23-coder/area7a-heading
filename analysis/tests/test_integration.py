"""Tests for cue-integration predictions and measured weights.

Covers the Bayesian optimal threshold and weight predictions,
IntegrationResult diagnostics (improvement_ratio, optimality_ratio),
neuronal weight estimation from tuning curves, visual weight recovery
from PSE shifts, and PSE-by-conflict psychometric fitting.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from seven_a.integration import (
    IntegrationResult,
    measured_visual_weight,
    neuronal_weights,
    predicted_threshold,
    predicted_weights,
    pse_by_conflict,
    session_integration,
)


# ------------------------------------------------------------------
# predicted_threshold
# ------------------------------------------------------------------
class TestPredictedThreshold:
    """Bayesian-optimal combined threshold sigma_pred."""

    def test_equal_cues(self):
        """Two equal-precision cues: sigma_pred = sigma / sqrt(2)."""
        sigma = 10.0
        result = predicted_threshold(sigma, sigma)
        expected = sigma / np.sqrt(2)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_equal_cues_sqrt2_factor(self):
        """The 0.707 optimum: combined is ~70.7% of either single cue."""
        sigma = 8.0
        result = predicted_threshold(sigma, sigma)
        assert result / sigma == pytest.approx(1.0 / np.sqrt(2), rel=1e-10)

    def test_one_cue_much_better(self):
        """Combined threshold is dominated by the better cue."""
        sigma_good = 2.0
        sigma_bad = 200.0
        result = predicted_threshold(sigma_good, sigma_bad)
        # Should be very close to the good cue alone
        assert result < sigma_good
        assert result == pytest.approx(sigma_good, abs=0.02)

    def test_asymmetric(self):
        """Check the formula for two different thresholds."""
        sa, sb = 6.0, 8.0
        expected = np.sqrt((sa**2 * sb**2) / (sa**2 + sb**2))
        assert predicted_threshold(sa, sb) == pytest.approx(expected, rel=1e-10)

    def test_symmetric(self):
        """predicted_threshold(a, b) == predicted_threshold(b, a)."""
        assert predicted_threshold(3.0, 7.0) == pytest.approx(
            predicted_threshold(7.0, 3.0), rel=1e-12
        )

    def test_negative_returns_nan(self):
        assert np.isnan(predicted_threshold(-1.0, 5.0))

    def test_zero_returns_nan(self):
        assert np.isnan(predicted_threshold(0.0, 5.0))
        assert np.isnan(predicted_threshold(5.0, 0.0))

    def test_nan_input_returns_nan(self):
        assert np.isnan(predicted_threshold(np.nan, 5.0))
        assert np.isnan(predicted_threshold(5.0, np.nan))

    def test_inf_returns_nan(self):
        assert np.isnan(predicted_threshold(np.inf, 5.0))


# ------------------------------------------------------------------
# predicted_weights
# ------------------------------------------------------------------
class TestPredictedWeights:
    """Bayesian-optimal cue weights w_a, w_b."""

    def test_equal_cues(self):
        """Equal precisions -> both weights = 0.5."""
        wa, wb = predicted_weights(5.0, 5.0)
        assert wa == pytest.approx(0.5, abs=1e-12)
        assert wb == pytest.approx(0.5, abs=1e-12)

    def test_weights_sum_to_one(self):
        """Weights must sum to 1 for any valid input pair."""
        for sa, sb in [(3.0, 7.0), (1.0, 100.0), (10.0, 10.0)]:
            wa, wb = predicted_weights(sa, sb)
            assert wa + wb == pytest.approx(1.0, abs=1e-12)

    def test_reliable_cue_dominates(self):
        """The cue with smaller sigma gets the larger weight."""
        wa, wb = predicted_weights(2.0, 20.0)
        # w_a = sigma_b^2 / (sigma_a^2 + sigma_b^2) = 400/404 ~ 0.99
        assert wa > 0.98
        assert wb < 0.02

    def test_weight_formula(self):
        """Verify the formula w_a = sigma_b^2 / (sigma_a^2 + sigma_b^2)."""
        sa, sb = 4.0, 6.0
        wa, wb = predicted_weights(sa, sb)
        assert wa == pytest.approx(sb**2 / (sa**2 + sb**2), rel=1e-10)
        assert wb == pytest.approx(sa**2 / (sa**2 + sb**2), rel=1e-10)

    def test_invalid_returns_nan_pair(self):
        wa, wb = predicted_weights(-1.0, 5.0)
        assert np.isnan(wa)
        assert np.isnan(wb)

    def test_nan_input(self):
        wa, wb = predicted_weights(np.nan, 5.0)
        assert np.isnan(wa)
        assert np.isnan(wb)


# ------------------------------------------------------------------
# IntegrationResult
# ------------------------------------------------------------------
class TestIntegrationResult:
    """Diagnostics: improvement_ratio and optimality_ratio."""

    def test_improvement_ratio_below_one(self):
        """When combining helps, ratio < 1."""
        r = IntegrationResult(
            sigma_ves=10.0,
            sigma_vis=10.0,
            sigma_com=7.0,  # better than either alone
            sigma_pred=predicted_threshold(10.0, 10.0),
            w_vis_pred=0.5,
        )
        assert r.improvement_ratio < 1.0

    def test_improvement_ratio_uses_best_single(self):
        """Ratio = sigma_com / min(sigma_ves, sigma_vis)."""
        r = IntegrationResult(
            sigma_ves=8.0,
            sigma_vis=12.0,
            sigma_com=6.0,
            sigma_pred=predicted_threshold(8.0, 12.0),
            w_vis_pred=0.5,
        )
        assert r.improvement_ratio == pytest.approx(6.0 / 8.0, rel=1e-10)

    def test_improvement_ratio_above_one_when_worse(self):
        """Combining can hurt if the model is wrong or noise is high."""
        r = IntegrationResult(
            sigma_ves=5.0, sigma_vis=5.0, sigma_com=8.0,
            sigma_pred=3.0, w_vis_pred=0.5,
        )
        assert r.improvement_ratio > 1.0

    def test_optimality_ratio_one_when_perfect(self):
        """optimality_ratio = 1 when measured == predicted."""
        sp = predicted_threshold(10.0, 10.0)
        r = IntegrationResult(
            sigma_ves=10.0, sigma_vis=10.0, sigma_com=sp,
            sigma_pred=sp, w_vis_pred=0.5,
        )
        assert r.optimality_ratio == pytest.approx(1.0, rel=1e-10)

    def test_optimality_ratio_above_one_suboptimal(self):
        """Sub-optimal combination has ratio > 1."""
        sp = predicted_threshold(10.0, 10.0)
        r = IntegrationResult(
            sigma_ves=10.0, sigma_vis=10.0, sigma_com=sp + 2.0,
            sigma_pred=sp, w_vis_pred=0.5,
        )
        assert r.optimality_ratio > 1.0

    def test_optimality_ratio_nan_when_pred_invalid(self):
        r = IntegrationResult(
            sigma_ves=10.0, sigma_vis=10.0, sigma_com=7.0,
            sigma_pred=np.nan, w_vis_pred=np.nan,
        )
        assert np.isnan(r.optimality_ratio)


# ------------------------------------------------------------------
# session_integration
# ------------------------------------------------------------------
class TestSessionIntegration:
    """Integration of predicted_threshold/predicted_weights into a result."""

    def test_bundles_correctly(self):
        r = session_integration(10.0, 8.0, 6.0, w_vis_measured=0.55)
        assert r.sigma_ves == 10.0
        assert r.sigma_vis == 8.0
        assert r.sigma_com == 6.0
        assert r.sigma_pred == pytest.approx(predicted_threshold(10.0, 8.0))
        # w_vis_pred = predicted_weights(sigma_vis, sigma_ves)[0]
        expected_w, _ = predicted_weights(8.0, 10.0)
        assert r.w_vis_pred == pytest.approx(expected_w)
        assert r.w_vis_measured == 0.55

    def test_default_measured_weight_is_nan(self):
        r = session_integration(5.0, 5.0, 3.5)
        assert np.isnan(r.w_vis_measured)


# ------------------------------------------------------------------
# neuronal_weights
# ------------------------------------------------------------------
class TestNeuronalWeights:
    """Estimating cue weights from tuning curves via least-squares."""

    def test_equal_average(self):
        """Combined = average of single cues -> both weights = 0.5."""
        rng = np.random.default_rng(42)
        ves = rng.uniform(5, 50, 20)
        vis = rng.uniform(5, 50, 20)
        com = (ves + vis) / 2.0
        result = neuronal_weights(ves, vis, com)
        assert result["w_ves"] == pytest.approx(0.5, abs=1e-8)
        assert result["w_vis"] == pytest.approx(0.5, abs=1e-8)
        assert result["r_squared"] == pytest.approx(1.0, abs=1e-8)

    def test_combined_equals_one_cue(self):
        """Combined = vestibular only -> w_ves = 1, w_vis = 0."""
        rng = np.random.default_rng(99)
        ves = rng.uniform(10, 80, 15)
        vis = rng.uniform(10, 80, 15)
        com = ves.copy()
        result = neuronal_weights(ves, vis, com)
        assert result["w_ves"] == pytest.approx(1.0, abs=1e-6)
        assert result["w_vis"] == pytest.approx(0.0, abs=1e-6)

    def test_r_squared_perfect(self):
        """Perfect linear combination gives r_squared = 1."""
        rng = np.random.default_rng(7)
        ves = rng.uniform(0, 40, 25)
        vis = rng.uniform(0, 40, 25)
        com = 0.3 * ves + 0.7 * vis
        result = neuronal_weights(ves, vis, com)
        assert result["r_squared"] == pytest.approx(1.0, abs=1e-8)
        assert result["w_ves"] == pytest.approx(0.3, abs=1e-6)
        assert result["w_vis"] == pytest.approx(0.7, abs=1e-6)

    def test_normalised_weights_sum_to_one(self):
        """Normalised weights should sum to 1."""
        rng = np.random.default_rng(11)
        ves = rng.uniform(0, 50, 20)
        vis = rng.uniform(0, 50, 20)
        com = 0.4 * ves + 0.6 * vis + rng.normal(0, 0.5, 20)
        result = neuronal_weights(ves, vis, com)
        total = result["w_ves_norm"] + result["w_vis_norm"]
        assert total == pytest.approx(1.0, abs=1e-10)

    def test_force_nonnegative(self):
        """With force_nonnegative, weights are >= 0."""
        rng = np.random.default_rng(77)
        ves = rng.uniform(10, 50, 20)
        vis = rng.uniform(10, 50, 20)
        # Construct a combined that anti-correlates with vis to push w_vis negative
        com = 1.2 * ves - 0.3 * vis + rng.normal(0, 0.1, 20)
        result_free = neuronal_weights(ves, vis, com, force_nonnegative=False)
        result_nn = neuronal_weights(ves, vis, com, force_nonnegative=True)
        # Unconstrained should have a negative w_vis
        assert result_free["w_vis"] < 0
        # Constrained must be >= 0
        assert result_nn["w_ves"] >= 0
        assert result_nn["w_vis"] >= 0

    def test_too_few_points_returns_nan(self):
        """Fewer than 3 valid points -> all nan."""
        result = neuronal_weights([1.0, 2.0], [3.0, 4.0], [5.0, 6.0])
        assert np.isnan(result["w_ves"])
        assert np.isnan(result["r_squared"])

    def test_nan_entries_excluded(self):
        """NaN entries are dropped; remaining valid points still fit."""
        rng = np.random.default_rng(55)
        ves = rng.uniform(5, 50, 10).tolist()
        vis = rng.uniform(5, 50, 10).tolist()
        com = [0.5 * v + 0.5 * b for v, b in zip(ves, vis)]
        # Inject nans at different positions
        ves[0] = np.nan
        vis[3] = np.nan
        com[7] = np.nan
        result = neuronal_weights(ves, vis, com)
        # Should still produce finite results from the remaining 7 valid points
        assert np.isfinite(result["w_ves"])
        assert np.isfinite(result["r_squared"])


# ------------------------------------------------------------------
# measured_visual_weight
# ------------------------------------------------------------------
class TestMeasuredVisualWeight:
    """Recovering visual weight from PSE shifts across conflict levels."""

    def test_known_weight_recovery(self):
        """PSE = -delta * (w_vis - 0.5) should recover w_vis from the slope."""
        w_vis_true = 0.7
        deltas = np.array([-8.0, -4.0, 0.0, 4.0, 8.0])
        pses = -deltas * (w_vis_true - 0.5)
        w_vis_est, p_value = measured_visual_weight(deltas, pses)
        assert w_vis_est == pytest.approx(w_vis_true, abs=1e-10)
        # Perfect linear relationship => p-value ~ 0
        assert p_value < 0.01

    def test_weight_half_gives_zero_pse(self):
        """w_vis = 0.5 means no PSE shift, slope = 0."""
        deltas = np.array([-6.0, -3.0, 0.0, 3.0, 6.0])
        pses = np.zeros(5)
        w_vis, _ = measured_visual_weight(deltas, pses)
        assert w_vis == pytest.approx(0.5, abs=1e-10)

    def test_insufficient_data_returns_nan(self):
        """Fewer than 2 points or no range in delta -> nan."""
        w, p = measured_visual_weight([1.0], [2.0])
        assert np.isnan(w)
        assert np.isnan(p)

    def test_constant_delta_returns_nan(self):
        """All same delta value -> no regression possible."""
        w, p = measured_visual_weight([5.0, 5.0, 5.0], [1.0, 2.0, 3.0])
        assert np.isnan(w)
        assert np.isnan(p)

    def test_noisy_recovery(self):
        """With noise the weight should still be approximately correct."""
        rng = np.random.default_rng(123)
        w_vis_true = 0.65
        deltas = np.linspace(-10, 10, 20)
        pses = -deltas * (w_vis_true - 0.5) + rng.normal(0, 0.3, 20)
        w_est, p_val = measured_visual_weight(deltas, pses)
        assert w_est == pytest.approx(w_vis_true, abs=0.1)
        assert p_val < 0.05

    def test_nan_entries_handled(self):
        """NaN entries in deltas or pses are excluded."""
        deltas = [np.nan, -4.0, 0.0, 4.0, 8.0]
        pses = [-0.5, 0.8, 0.0, -0.8, np.nan]
        w, p = measured_visual_weight(deltas, pses)
        # Only 3 valid points remain (indices 1, 2, 3), still enough
        assert np.isfinite(w)


# ------------------------------------------------------------------
# pse_by_conflict
# ------------------------------------------------------------------
class TestPseByConflict:
    """Fitting psychometric functions per conflict level."""

    @staticmethod
    def _generate_conflict_data(
        headings: np.ndarray,
        w_vis: float,
        deltas: list[float],
        n_repeat: int = 30,
        sigma: float = 5.0,
        rng: np.random.Generator | None = None,
    ):
        """Generate synthetic cue-conflict psychometric data.

        For each conflict level delta and each heading h, the observer
        with visual weight w_vis has an effective PSE shift. We simulate
        binary choices from a cumulative-Gaussian observer.
        """
        if rng is None:
            rng = np.random.default_rng(314)
        all_h, all_r, all_d = [], [], []
        for delta in deltas:
            pse = -delta * (w_vis - 0.5)
            for h in headings:
                p_right = stats.norm.cdf((h - pse) / sigma)
                choices = rng.binomial(1, p_right, size=n_repeat)
                all_h.extend([h] * n_repeat)
                all_r.extend(choices.tolist())
                all_d.extend([delta] * n_repeat)
        return np.array(all_h), np.array(all_r, dtype=float), np.array(all_d)

    def test_pse_shifts_linearly(self):
        """PSEs should shift approximately linearly with delta."""
        headings = np.array([-16, -8, -4, -2, 0, 2, 4, 8, 16], dtype=float)
        w_vis = 0.7
        deltas = [-8.0, -4.0, 0.0, 4.0, 8.0]
        h, r, d = self._generate_conflict_data(headings, w_vis, deltas, n_repeat=60)
        result = pse_by_conflict(h, r, d, fit_lapse=False)
        assert len(result) == len(deltas)
        # Check that the PSE at delta=0 is near 0
        assert result[0.0] == pytest.approx(0.0, abs=1.5)

    def test_pse_direction(self):
        """Positive delta should shift PSE in the predicted direction."""
        headings = np.array([-16, -8, -4, -2, 0, 2, 4, 8, 16], dtype=float)
        w_vis = 0.8
        deltas = [-8.0, 0.0, 8.0]
        h, r, d = self._generate_conflict_data(headings, w_vis, deltas, n_repeat=80)
        result = pse_by_conflict(h, r, d, fit_lapse=False)
        # With w_vis > 0.5, PSE = -delta*(w_vis - 0.5), so larger delta -> more negative PSE
        assert result[8.0] < result[0.0]
        assert result[0.0] < result[-8.0]

    def test_too_few_trials_skipped(self):
        """Conflict levels with fewer than 8 trials are skipped."""
        headings = np.array([0.0, 1.0, 2.0])
        rightward = np.array([0.0, 0.5, 1.0])
        deltas = np.array([0.0, 0.0, 0.0])  # only 3 trials at delta=0
        result = pse_by_conflict(headings, rightward, deltas)
        assert len(result) == 0
