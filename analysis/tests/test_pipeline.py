"""Ground-truth tests. Run with ``python -m pytest analysis/tests -q`` from the repo root."""

from __future__ import annotations

import numpy as np
import pytest

from seven_a import behavior, integration, neurometrics, synth


# --------------------------------------------------------------------------------------
# Psychometric fitting
# --------------------------------------------------------------------------------------
def test_psychometric_recovers_threshold_and_bias():
    rng = np.random.default_rng(1)
    h = np.repeat(synth.HEADINGS_HD, 400)
    p = behavior.cumulative_gaussian(h, mu=1.5, sigma=4.0, lapse=0.02)
    right = (rng.random(h.size) < p).astype(float)

    fit = behavior.fit_psychometric(h, right)

    assert fit.converged
    assert fit.sigma == pytest.approx(4.0, abs=0.5)
    assert fit.mu == pytest.approx(1.5, abs=0.4)


def test_psychometric_bootstrap_ci_brackets_estimate():
    rng = np.random.default_rng(2)
    h = np.repeat(synth.HEADINGS_HD, 120)
    right = (rng.random(h.size) < behavior.cumulative_gaussian(h, 0.0, 3.0)).astype(float)

    fit = behavior.fit_psychometric(h, right, n_boot=150, rng=rng)

    lo, hi = fit.ci["sigma"]
    assert lo < fit.sigma < hi


# --------------------------------------------------------------------------------------
# The choice-coding resolver -- the part most likely to silently corrupt everything
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("scheme", ["absolute", "stimulus_relative"])
@pytest.mark.parametrize("high_is_right", [True, False])
def test_resolver_recovers_encoding(scheme, high_is_right):
    rng = np.random.default_rng(3)
    h = np.repeat(synth.HEADINGS_HD, 200)
    right = (rng.random(h.size) < behavior.cumulative_gaussian(h, 0.0, 4.0, 0.02)).astype(float)
    raw = synth.encode_choice(
        h, right, scheme=scheme, values=(0.0, 5.0), high_is_right=high_is_right, rng=rng
    )

    coding = behavior.resolve_choice_coding(h, raw)

    assert coding.scheme == scheme
    assert coding.high_is_right == high_is_right

    recovered = coding.apply(h, raw)
    defined = np.isfinite(recovered)
    # Zero-heading trials are genuinely unrecoverable under stimulus-relative coding.
    if scheme == "stimulus_relative":
        assert not defined[np.isclose(h, 0.0)].any()
    assert np.array_equal(recovered[defined], right[defined])


def test_resolver_flags_bell_shape_for_wrong_reading():
    """The absolute reading of relative data should look symmetric, not monotonic.

    This is the diagnostic JP described: a Gaussian instead of a sigmoid.
    """
    rng = np.random.default_rng(4)
    h = np.repeat(synth.HEADINGS_HD, 200)
    right = (rng.random(h.size) < behavior.cumulative_gaussian(h, 0.0, 4.0)).astype(float)
    raw = synth.encode_choice(h, right, scheme="stimulus_relative", rng=rng)

    coding = behavior.resolve_choice_coding(h, raw)
    absolute = [c for c in coding.diagnostics["candidates"] if c["scheme"] == "absolute"]

    # Reading it as absolute gives a bell shape: strong dependence on |heading|, none on heading.
    assert max(abs(c["symmetry"]) for c in absolute) > 0.8
    assert min(abs(c["monotonicity"]) for c in absolute) < 0.5
    # And the selected (relative) reading is monotonic.
    assert coding.diagnostics["selected"]["monotonicity"] > 0.8


def test_resolver_rejects_non_binary_choice():
    h = np.repeat(synth.HEADINGS_HD, 10)
    raw = np.tile([0.0, 5.0, -5.0], h.size // 3 + 1)[: h.size]

    with pytest.raises(ValueError, match="exactly 2 distinct"):
        behavior.resolve_choice_coding(h, raw)


# --------------------------------------------------------------------------------------
# ROC / neurometrics
# --------------------------------------------------------------------------------------
def test_roc_auc_known_cases():
    assert neurometrics.roc_auc([2, 3, 4], [0, 1]) == pytest.approx(1.0)
    assert neurometrics.roc_auc([0, 1], [2, 3, 4]) == pytest.approx(0.0)
    # Full ties give exactly chance.
    assert neurometrics.roc_auc([1, 1, 1], [1, 1, 1]) == pytest.approx(0.5)


def test_neurometric_is_symmetric_and_ordered():
    rng = np.random.default_rng(5)
    h = np.repeat(synth.HEADINGS_HD, 60)
    rates = rng.poisson(np.clip(3.0 * np.exp(0.08 * h), 0.05, None)).astype(float)

    res = neurometrics.neurometric_function(h, rates)

    assert res.prefers_right
    ok = np.isfinite(res.p_right)
    # Antineuron construction forces p(-theta) = 1 - p(theta).
    assert np.allclose(res.p_right[ok], 1.0 - res.p_right[ok][::-1], atol=1e-9)
    assert res.p_right[-1] > 0.5 > res.p_right[0]
    assert np.isfinite(res.threshold)


def test_neurometric_threshold_worse_for_weaker_tuning():
    rng = np.random.default_rng(6)
    h = np.repeat(synth.HEADINGS_HD, 80)
    strong = rng.poisson(np.clip(5.0 * np.exp(0.15 * h), 0.05, None)).astype(float)
    weak = rng.poisson(np.clip(5.0 * np.exp(0.02 * h), 0.05, None)).astype(float)

    t_strong = neurometrics.neurometric_function(h, strong).threshold
    t_weak = neurometrics.neurometric_function(h, weak).threshold

    assert t_strong < t_weak


# --------------------------------------------------------------------------------------
# Choice probability
# --------------------------------------------------------------------------------------
def test_cp_at_chance_when_rates_independent_of_choice():
    rng = np.random.default_rng(7)
    h = np.repeat(synth.HEADINGS_HD, 60)
    right = (rng.random(h.size) < behavior.cumulative_gaussian(h, 0.0, 4.0)).astype(float)
    rates = rng.poisson(np.clip(3.0 * np.exp(0.08 * h), 0.05, None)).astype(float)

    cp = neurometrics.choice_probability(h, rates, right, n_perm=500, rng=rng)

    assert cp.cp == pytest.approx(0.5, abs=0.06)
    assert cp.p_value > 0.05


def test_cp_above_chance_when_rates_track_choice():
    rng = np.random.default_rng(8)
    h = np.repeat(synth.HEADINGS_HD, 60)
    right = (rng.random(h.size) < behavior.cumulative_gaussian(h, 0.0, 4.0)).astype(float)
    base = 3.0 * np.exp(0.08 * h) * np.exp(0.5 * np.where(right == 1, 1.0, -1.0))
    rates = rng.poisson(np.clip(base, 0.05, None)).astype(float)

    cp = neurometrics.choice_probability(h, rates, right, n_perm=500, rng=rng)

    assert cp.cp > 0.6
    assert cp.p_value < 0.01


def test_cp_permutation_p_is_never_zero():
    rng = np.random.default_rng(9)
    h = np.repeat(synth.HEADINGS_HD, 40)
    right = (rng.random(h.size) < 0.5).astype(float)
    rates = np.where(right == 1, 10.0, 1.0) + rng.normal(0, 0.1, h.size)

    cp = neurometrics.choice_probability(h, rates, right, n_perm=200, rng=rng)

    assert cp.p_value > 0


# --------------------------------------------------------------------------------------
# Tuning descriptors
# --------------------------------------------------------------------------------------
def test_ddi_higher_for_stronger_modulation():
    rng = np.random.default_rng(10)
    h = np.repeat(synth.HEADINGS_HD, 40)
    strong = 5.0 + 0.4 * h + rng.normal(0, 0.5, h.size)
    flat = 5.0 + rng.normal(0, 0.5, h.size)

    assert neurometrics.discrimination_index(h, strong) > neurometrics.discrimination_index(h, flat)


def test_congruency_labels():
    rng = np.random.default_rng(11)
    h = np.repeat(synth.HEADINGS_HD, 50)
    up = 5.0 + 0.3 * h + rng.normal(0, 0.5, h.size)
    down = 5.0 - 0.3 * h + rng.normal(0, 0.5, h.size)
    flat = 5.0 + rng.normal(0, 0.5, h.size)

    assert neurometrics.congruency(h, up, h, up)["label"] == "congruent"
    assert neurometrics.congruency(h, up, h, down)["label"] == "opposite"
    assert neurometrics.congruency(h, up, h, flat)["label"] == "untuned"


def test_partial_correlation_separates_choice_from_heading():
    """A unit driven purely by heading should show no residual choice correlation."""
    rng = np.random.default_rng(12)
    h = np.repeat(synth.HEADINGS_HD, 100)
    right = (rng.random(h.size) < behavior.cumulative_gaussian(h, 0.0, 4.0)).astype(float)
    rates = 3.0 + 0.3 * h + rng.normal(0, 1.0, h.size)

    pc = neurometrics.partial_correlations(rates, h, right)

    assert abs(pc["r_choice_given_heading"]) < 0.1
    assert pc["r_heading_given_choice"] > 0.4


# --------------------------------------------------------------------------------------
# Cue integration
# --------------------------------------------------------------------------------------
def test_optimal_prediction_matches_eric_example_session():
    """Eric's example session: ves 4.05, vis 3.50, com 2.65 (image6 of 1D_HD_plots.docx)."""
    assert integration.predicted_threshold(4.05, 3.50) == pytest.approx(2.65, abs=0.01)


def test_equal_reliability_gives_sqrt2_improvement_and_equal_weights():
    assert integration.predicted_threshold(4.0, 4.0) == pytest.approx(4.0 / np.sqrt(2))
    w_a, w_b = integration.predicted_weights(4.0, 4.0)
    assert w_a == pytest.approx(0.5)
    assert w_b == pytest.approx(0.5)


def test_more_reliable_cue_gets_more_weight():
    w_vis, w_ves = integration.predicted_weights(sigma_a=2.0, sigma_b=6.0)
    assert w_vis > w_ves
    assert w_vis + w_ves == pytest.approx(1.0)


def test_measured_weight_recovered_from_pse_shifts():
    deltas = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
    w_true = 0.7
    pses = -deltas * (w_true - 0.5)

    w_hat, _ = integration.measured_visual_weight(deltas, pses)

    assert w_hat == pytest.approx(w_true, abs=1e-6)


# --------------------------------------------------------------------------------------
# End-to-end
# --------------------------------------------------------------------------------------
def test_end_to_end_session_recovers_thresholds_and_optimality():
    sess = synth.simulate_session(n_units=6, n_reps=90, rng=np.random.default_rng(13))

    fits = {}
    for mod in ("ves", "vis", "com"):
        coding = behavior.resolve_choice_coding(sess.headings[mod], sess.raw_choice[mod])
        assert coding.scheme == "stimulus_relative"
        right = coding.apply(sess.headings[mod], sess.raw_choice[mod])
        keep = np.isfinite(right)
        fits[mod] = behavior.fit_psychometric(sess.headings[mod][keep], right[keep])

    for mod in ("ves", "vis", "com"):
        assert fits[mod].sigma == pytest.approx(sess.truth[f"sigma_{mod}"], rel=0.3)

    res = integration.session_integration(
        fits["ves"].sigma, fits["vis"].sigma, fits["com"].sigma
    )
    assert res.optimality_ratio == pytest.approx(1.0, abs=0.35)
    assert res.improvement_ratio < 1.0
