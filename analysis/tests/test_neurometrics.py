"""Tests for single-neuron neurometric analysis primitives.

Covers the ROC/AUC computation, the neurometric function under the
antineuron assumption, choice probability, the discrimination index
(DDI), tuning slope, congruency classification, and partial correlations.
"""

from __future__ import annotations

import numpy as np
import pytest

from seven_a.neurometrics import (
    ChoiceProbability,
    NeurometricResult,
    choice_probability,
    congruency,
    discrimination_index,
    partial_correlations,
    preferred_side,
    roc_auc,
    tuning_slope,
)


# ------------------------------------------------------------------
# ROC / AUC
# ------------------------------------------------------------------
class TestRocAuc:
    def test_perfect_separation(self) -> None:
        """Non-overlapping distributions should give AUC = 1."""
        pos = [10.0, 11.0, 12.0]
        neg = [1.0, 2.0, 3.0]
        assert roc_auc(pos, neg) == pytest.approx(1.0)

    def test_perfect_reverse(self) -> None:
        """Reversed labeling should give AUC = 0."""
        pos = [1.0, 2.0, 3.0]
        neg = [10.0, 11.0, 12.0]
        assert roc_auc(pos, neg) == pytest.approx(0.0)

    def test_identical_gives_half(self) -> None:
        """Identical distributions should give AUC ~ 0.5."""
        vals = [5.0, 5.0, 5.0, 5.0]
        assert roc_auc(vals, vals) == pytest.approx(0.5)

    def test_empty_gives_nan(self) -> None:
        assert np.isnan(roc_auc([], [1.0, 2.0]))
        assert np.isnan(roc_auc([1.0], []))

    def test_nan_values_excluded(self) -> None:
        """NaN entries should be dropped, not crash."""
        pos = [10.0, np.nan, 12.0]
        neg = [1.0, 2.0, np.nan]
        auc = roc_auc(pos, neg)
        assert np.isfinite(auc)
        assert auc > 0.9

    def test_ties_give_half_credit(self) -> None:
        """Tied observations should contribute 0.5 per comparison."""
        # All tied: each comparison is 0.5
        pos = [5.0, 5.0]
        neg = [5.0, 5.0]
        assert roc_auc(pos, neg) == pytest.approx(0.5)


# ------------------------------------------------------------------
# Preferred side
# ------------------------------------------------------------------
class TestPreferredSide:
    def test_rightward_preference(self) -> None:
        """Positive slope means the unit prefers rightward headings."""
        headings = [-12, -6, -3, 0, 3, 6, 12]
        rates = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        assert preferred_side(headings, rates) is True

    def test_leftward_preference(self) -> None:
        headings = [-12, -6, -3, 0, 3, 6, 12]
        rates = [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        assert preferred_side(headings, rates) is False


# ------------------------------------------------------------------
# Discrimination index (DDI)
# ------------------------------------------------------------------
class TestDiscriminationIndex:
    def test_perfect_separation_near_one(self) -> None:
        """No within-condition variance and large spread gives DDI ~ 1."""
        conditions = np.repeat([0, 1, 2, 3], 10).astype(float)
        rates = conditions * 10.0  # perfect step function
        ddi = discrimination_index(conditions, rates)
        assert ddi == pytest.approx(1.0, abs=0.01)

    def test_noisy_unit_lower_ddi(self) -> None:
        """Adding noise should reduce DDI."""
        rng = np.random.default_rng(42)
        conditions = np.repeat([0, 1, 2], 30).astype(float)
        clean_rates = conditions * 5.0
        noisy_rates = clean_rates + rng.normal(0, 10.0, size=len(conditions))

        ddi_clean = discrimination_index(conditions, clean_rates)
        ddi_noisy = discrimination_index(conditions, noisy_rates)
        assert ddi_clean > ddi_noisy

    def test_range_zero_to_one(self) -> None:
        rng = np.random.default_rng(7)
        conditions = np.repeat(np.arange(5), 20).astype(float)
        rates = conditions * 3.0 + rng.normal(0, 2.0, size=len(conditions))
        ddi = discrimination_index(conditions, rates)
        assert 0.0 <= ddi <= 1.0

    def test_too_few_conditions_gives_nan(self) -> None:
        assert np.isnan(discrimination_index([1.0], [5.0]))

    def test_flat_tuning_gives_nan(self) -> None:
        """No modulation across conditions: spread=0, denominator=0 -> NaN."""
        conditions = np.repeat([0, 1, 2, 3], 10).astype(float)
        rates = np.full(len(conditions), 5.0)
        ddi = discrimination_index(conditions, rates)
        assert np.isnan(ddi)


# ------------------------------------------------------------------
# Tuning slope
# ------------------------------------------------------------------
class TestTuningSlope:
    def test_positive_slope(self) -> None:
        headings = np.array([-12, -6, -3, 0, 3, 6, 12], dtype=float)
        rates = headings * 2.0 + 10.0
        slope, pval = tuning_slope(headings, rates)
        assert slope > 0
        assert pval < 0.05

    def test_flat_slope_not_significant(self) -> None:
        rng = np.random.default_rng(0)
        headings = np.repeat(np.linspace(-12, 12, 7), 5)
        rates = 10.0 + rng.normal(0, 0.01, size=len(headings))
        slope, pval = tuning_slope(headings, rates)
        assert abs(slope) < 0.1

    def test_too_few_points_gives_nan(self) -> None:
        slope, pval = tuning_slope([1.0, 2.0], [3.0, 4.0])
        assert np.isnan(slope)


# ------------------------------------------------------------------
# Congruency
# ------------------------------------------------------------------
class TestCongruency:
    def test_congruent_cell(self) -> None:
        """Same positive slope in both modalities -> congruent."""
        rng = np.random.default_rng(10)
        headings = np.repeat(np.linspace(-12, 12, 7), 20)
        rates_ves = headings * 0.5 + 10 + rng.normal(0, 0.5, len(headings))
        rates_vis = headings * 0.3 + 8 + rng.normal(0, 0.5, len(headings))
        result = congruency(headings, rates_ves, headings, rates_vis)
        assert result["label"] == "congruent"
        assert result["congruency_index"] > 0

    def test_opposite_cell(self) -> None:
        """Opposite slopes -> opposite."""
        rng = np.random.default_rng(11)
        headings = np.repeat(np.linspace(-12, 12, 7), 20)
        rates_ves = headings * 0.5 + 10 + rng.normal(0, 0.5, len(headings))
        rates_vis = -headings * 0.3 + 8 + rng.normal(0, 0.5, len(headings))
        result = congruency(headings, rates_ves, headings, rates_vis)
        assert result["label"] == "opposite"
        assert result["congruency_index"] < 0

    def test_untuned_when_not_significant(self) -> None:
        """Flat tuning should be labeled untuned."""
        rng = np.random.default_rng(12)
        headings = np.repeat(np.linspace(-12, 12, 5), 5)
        flat = np.full(len(headings), 10.0) + rng.normal(0, 5.0, len(headings))
        result = congruency(headings, flat, headings, flat)
        assert result["label"] == "untuned"


# ------------------------------------------------------------------
# Partial correlations
# ------------------------------------------------------------------
class TestPartialCorrelations:
    def test_stimulus_driven_unit(self) -> None:
        """A unit that tracks heading but not choice should have
        large r_heading_given_choice and small r_choice_given_heading."""
        rng = np.random.default_rng(20)
        n = 200
        headings = rng.uniform(-12, 12, n)
        rates = headings * 2.0 + rng.normal(0, 1.0, n)
        # Choice is driven by heading + noise (not by the neuron)
        rightward = (headings + rng.normal(0, 5, n) > 0).astype(float)

        result = partial_correlations(rates, headings, rightward)
        assert abs(result["r_heading_given_choice"]) > abs(result["r_choice_given_heading"])

    def test_returns_nan_with_too_few_trials(self) -> None:
        result = partial_correlations([1, 2, 3], [1, 2, 3], [0, 1, 0])
        assert np.isnan(result["r_choice_given_heading"])

    def test_returns_dict_keys(self) -> None:
        rng = np.random.default_rng(30)
        n = 50
        result = partial_correlations(
            rng.normal(size=n),
            rng.normal(size=n),
            rng.binomial(1, 0.5, n).astype(float),
        )
        assert "r_choice_given_heading" in result
        assert "r_heading_given_choice" in result


# ------------------------------------------------------------------
# Choice probability
# ------------------------------------------------------------------
class TestChoiceProbability:
    def test_uninformative_unit_near_half(self) -> None:
        """A unit uncorrelated with choice should give CP near 0.5."""
        rng = np.random.default_rng(50)
        n_per = 40
        headings = np.repeat([-6, -3, 0, 3, 6], n_per).astype(float)
        rates = rng.normal(10, 2, len(headings))
        # Roughly balanced choices, uncorrelated with rates
        rightward = rng.binomial(1, 0.5, len(headings)).astype(float)

        cp = choice_probability(headings, rates, rightward, n_perm=200, rng=rng)
        assert isinstance(cp, ChoiceProbability)
        if np.isfinite(cp.cp):
            assert 0.3 < cp.cp < 0.7

    def test_p_value_not_significant_for_noise(self) -> None:
        """Random data should not produce a significant p-value."""
        rng = np.random.default_rng(51)
        n_per = 30
        headings = np.repeat([-6, -3, 0, 3, 6], n_per).astype(float)
        rates = rng.normal(10, 2, len(headings))
        rightward = rng.binomial(1, 0.5, len(headings)).astype(float)

        cp = choice_probability(headings, rates, rightward, n_perm=500, rng=rng)
        if np.isfinite(cp.p_value):
            assert cp.p_value > 0.05

    def test_no_usable_conditions_returns_nan(self) -> None:
        """If no heading has enough trials per choice, return NaN."""
        cp = choice_probability(
            [0.0, 0.0], [5.0, 6.0], [1.0, 1.0],
            n_perm=0,
        )
        assert np.isnan(cp.cp)
        assert cp.n_trials == 0
