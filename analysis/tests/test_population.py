"""Tests for population decoding, spike windowing, and the corrected Gu-2008 primitives."""

from __future__ import annotations

import numpy as np
import pytest

from seven_a import integration, neurometrics, population, spikes, synth


# --------------------------------------------------------------------------------------
# Spike windowing
# --------------------------------------------------------------------------------------
def test_middle_window_matches_gu_convention():
    """HD trial: motion runs 0.5-2.5 s, so the middle 1 s is 1.0-2.0 s."""
    assert spikes.middle_window() == pytest.approx((1.0, 2.0))


def test_rate_in_window_counts_only_inside():
    ts = [0.1, 0.9, 1.2, 1.5, 1.9, 2.4]
    # Window is 1 s wide, so rate == count.
    assert spikes.rate_in_window(ts, 1.0, 2.0) == pytest.approx(3.0)
    assert spikes.rate_in_window(ts, 0.0, 0.5) == pytest.approx(2.0)


def test_rate_in_window_scales_by_duration():
    ts = [1.1, 1.2, 1.3, 1.4]
    assert spikes.rate_in_window(ts, 1.0, 1.5) == pytest.approx(8.0)


def test_rate_in_window_rejects_empty_interval():
    assert np.isnan(spikes.rate_in_window([1.0], 2.0, 1.0))


def test_psth_recovers_constant_rate():
    rng = np.random.default_rng(0)
    rate_hz = 20.0
    trials = [rng.uniform(0, 2.9, rng.poisson(rate_hz * 2.9)) for _ in range(200)]

    centres, rate = spikes.psth(trials, 0.0, 2.9)

    # Ignore edges, where the smoothing kernel runs off the end.
    interior = (centres > 0.3) & (centres < 2.6)
    assert np.mean(rate[interior]) == pytest.approx(rate_hz, rel=0.1)


def test_responsiveness_detects_excitation_and_suppression():
    rng = np.random.default_rng(1)
    onset = spikes.HD_MOTION_ONSET_S

    excited, suppressed = [], []
    for _ in range(40):
        pre = rng.uniform(onset - 0.25, onset, rng.poisson(2))
        strong = rng.uniform(onset, onset + 0.25, rng.poisson(12))
        excited.append(np.concatenate([pre, strong]))
        pre2 = rng.uniform(onset - 0.25, onset, rng.poisson(12))
        weak = rng.uniform(onset, onset + 0.25, rng.poisson(2))
        suppressed.append(np.concatenate([pre2, weak]))

    assert spikes.responsiveness(excited)[2] == "excitatory"
    assert spikes.responsiveness(suppressed)[2] == "suppressive"


def test_responsiveness_unresponsive_when_flat():
    rng = np.random.default_rng(2)
    onset = spikes.HD_MOTION_ONSET_S
    trials = [rng.uniform(onset - 0.25, onset + 0.25, rng.poisson(6)) for _ in range(40)]

    assert spikes.responsiveness(trials)[2] == "unresponsive"


# --------------------------------------------------------------------------------------
# Congruency index, corrected to Gu 2008's definition (CI = R_ves * R_vis)
# --------------------------------------------------------------------------------------
def test_congruency_index_is_product_of_correlations():
    rng = np.random.default_rng(3)
    h = np.repeat(synth.HEADINGS_HD, 60)
    ves = 5.0 + 0.30 * h + rng.normal(0, 1.0, h.size)
    vis = 5.0 + 0.20 * h + rng.normal(0, 1.0, h.size)

    res = neurometrics.congruency(h, ves, h, vis)

    assert res["congruency_index"] == pytest.approx(res["r_ves"] * res["r_vis"])
    assert res["label"] == "congruent"
    assert 0.0 < res["congruency_index"] <= 1.0


def test_congruency_index_negative_for_opposite_cells():
    rng = np.random.default_rng(4)
    h = np.repeat(synth.HEADINGS_HD, 60)
    ves = 5.0 + 0.30 * h + rng.normal(0, 1.0, h.size)
    vis = 5.0 - 0.30 * h + rng.normal(0, 1.0, h.size)

    res = neurometrics.congruency(h, ves, h, vis)

    assert res["label"] == "opposite"
    assert res["congruency_index"] < 0


def test_congruency_index_bounded_by_one():
    h = np.repeat(synth.HEADINGS_HD, 20)
    perfect = 0.3 * h
    res = neurometrics.congruency(h, perfect, h, perfect)
    assert res["congruency_index"] == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------------------
# Neuronal weights -- available without cue-conflict trials
# --------------------------------------------------------------------------------------
def test_neuronal_weights_recovered_from_tuning_curves():
    h = np.array(synth.HEADINGS_HD)
    ves = 4.0 + 0.5 * h
    vis = 3.0 - 0.2 * h
    com = 0.7 * ves + 0.4 * vis

    w = integration.neuronal_weights(ves, vis, com)

    assert w["w_ves"] == pytest.approx(0.7, abs=1e-6)
    assert w["w_vis"] == pytest.approx(0.4, abs=1e-6)
    assert w["r_squared"] == pytest.approx(1.0, abs=1e-6)


def test_neuronal_weights_normalised_sum_to_one():
    h = np.array(synth.HEADINGS_HD)
    ves = 4.0 + 0.5 * h
    vis = 3.0 - 0.2 * h
    com = 0.6 * ves + 0.2 * vis

    w = integration.neuronal_weights(ves, vis, com)

    assert w["w_ves_norm"] + w["w_vis_norm"] == pytest.approx(1.0)


def test_neuronal_weights_flag_bad_model_fit():
    """A combined response unrelated to either single cue should not look well described."""
    rng = np.random.default_rng(5)
    h = np.array(synth.HEADINGS_HD)
    ves = 4.0 + 0.5 * h
    vis = 3.0 - 0.2 * h
    com = rng.normal(5.0, 3.0, h.size)

    w = integration.neuronal_weights(ves, vis, com)

    assert w["r_squared"] < 0.5


# --------------------------------------------------------------------------------------
# Population decoding
# --------------------------------------------------------------------------------------
def _population(n_units, n_reps, slope, rng, cp_strength=0.0):
    """Simulated population with a common heading drive."""
    h = np.repeat(synth.HEADINGS_HD, n_reps)
    rng.shuffle(h)
    rates = np.empty((h.size, n_units))
    for u in range(n_units):
        lam = 3.0 * np.exp(slope * h * rng.choice([-1, 1]))
        rates[:, u] = rng.poisson(np.clip(lam, 0.05, None))
    return h, rates


def test_decode_heading_above_chance_and_yields_threshold():
    rng = np.random.default_rng(6)
    h, rates = _population(20, 60, 0.10, rng)

    res = population.decode_heading(rates, h, rng=rng)

    assert res.auc > 0.65
    assert res.n_units == 20
    assert np.isfinite(res.threshold)
    # Neurometric function should be ordered in heading.
    levels = sorted(res.per_heading)
    assert res.per_heading[levels[-1]] > res.per_heading[levels[0]]


def test_decode_heading_at_chance_for_unmodulated_population():
    rng = np.random.default_rng(7)
    h = np.repeat(synth.HEADINGS_HD, 60)
    rates = rng.poisson(3.0, (h.size, 15)).astype(float)

    res = population.decode_heading(rates, h, n_perm=60, rng=rng)

    assert res.auc == pytest.approx(0.5, abs=0.08)
    assert res.p_value > 0.05


def test_population_threshold_improves_with_more_units():
    rng = np.random.default_rng(8)
    h, rates = _population(32, 80, 0.08, rng)

    curve = population.threshold_vs_population_size(
        rates, h, sizes=[2, 32], n_draws=6, rng=rng
    )

    assert curve[32][0] < curve[2][0]


def test_decode_heading_validates_shapes():
    with pytest.raises(ValueError, match="10 trials but headings has 7"):
        population.decode_heading(np.zeros((10, 3)), np.zeros(7))
    with pytest.raises(ValueError, match="must be"):
        population.decode_heading(np.zeros(10), np.zeros(10))


def test_decode_choice_residualising_removes_stimulus_confound():
    """Rates driven only by heading must not yield choice decoding after residualising."""
    rng = np.random.default_rng(9)
    from seven_a import behavior

    h = np.repeat(synth.HEADINGS_HD, 80)
    right = (rng.random(h.size) < behavior.cumulative_gaussian(h, 0.0, 4.0)).astype(float)
    rates = np.column_stack(
        [rng.poisson(np.clip(3.0 * np.exp(0.12 * h), 0.05, None)) for _ in range(12)]
    ).astype(float)

    naive = population.decode_choice(rates, right, h, residualise=False, rng=rng)
    clean = population.decode_choice(rates, right, h, residualise=True, rng=rng)

    # Without residualising the decoder rides on the stimulus; with it, nothing is left.
    assert naive.auc > 0.6
    assert clean.auc == pytest.approx(0.5, abs=0.08)


def test_decode_choice_finds_genuine_choice_signal():
    rng = np.random.default_rng(10)
    from seven_a import behavior

    h = np.repeat(synth.HEADINGS_HD, 80)
    right = (rng.random(h.size) < behavior.cumulative_gaussian(h, 0.0, 4.0)).astype(float)
    signed = np.where(right == 1, 1.0, -1.0)
    rates = np.column_stack(
        [
            rng.poisson(np.clip(3.0 * np.exp(0.12 * h + 0.45 * signed), 0.05, None))
            for _ in range(12)
        ]
    ).astype(float)

    clean = population.decode_choice(rates, right, h, residualise=True, rng=rng)

    assert clean.auc > 0.65
