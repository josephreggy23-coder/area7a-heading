"""Single-neuron sensitivity, choice probability, and tuning descriptors.

Implements the primitives the three template papers are built on:

* ROC / AUC between two response distributions (exact, tie-aware).
* Neurometric functions under the antineuron assumption (Gu, DeAngelis & Angelaki 2007).
* Choice probability with a permutation test (Britten et al. 1996; Gu 2007).
* Discrimination index (DDI) and congruency between vestibular and visual tuning (Gu 2008).
* Partial correlations separating stimulus- from choice-driven activity (Zaidel et al. 2017).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy import stats

from .behavior import PsychometricFit, fit_psychometric

_MIN_TRIALS_PER_GROUP = 3


# --------------------------------------------------------------------------------------
# ROC
# --------------------------------------------------------------------------------------
def roc_auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """Area under the ROC curve for ``pos`` scoring higher than ``neg``.

    Computed from the Mann-Whitney U statistic on midranks, so ties contribute 0.5 rather
    than being broken arbitrarily. Returns NaN if either group is empty.
    """
    p = np.asarray(pos, dtype=float)
    n = np.asarray(neg, dtype=float)
    p = p[np.isfinite(p)]
    n = n[np.isfinite(n)]
    if p.size == 0 or n.size == 0:
        return np.nan
    ranks = stats.rankdata(np.concatenate([p, n]))
    u = float(np.sum(ranks[: p.size])) - p.size * (p.size + 1) / 2.0
    return u / (p.size * n.size)


# --------------------------------------------------------------------------------------
# Neurometric function
# --------------------------------------------------------------------------------------
@dataclass
class NeurometricResult:
    """Neurometric function and the threshold derived from it."""

    headings: np.ndarray
    p_right: np.ndarray
    n_trials: np.ndarray
    fit: PsychometricFit
    prefers_right: bool

    @property
    def threshold(self) -> float:
        return self.fit.sigma


def preferred_side(headings: Sequence[float], rates: Sequence[float]) -> bool:
    """True if the unit fires more for rightward (positive) headings.

    Uses the sign of the rate-vs-heading regression slope, which is the appropriate summary
    over the narrow (+/-12 deg) range used in the discrimination task, where tuning is close
    to linear.
    """
    x = np.asarray(headings, float)
    y = np.asarray(rates, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0:
        return True
    slope = float(np.polyfit(x[ok], y[ok], 1)[0])
    return slope >= 0


def neurometric_function(
    headings: Sequence[float],
    rates: Sequence[float],
    prefers_right: bool | None = None,
    fit_lapse: bool = False,
) -> NeurometricResult:
    """Neurometric function under the antineuron assumption.

    For each heading theta, the ideal observer compares the recorded neuron's response at
    theta against an antineuron with mirror-image tuning -- whose response distribution is
    taken to be the recorded neuron's response at -theta. So

        p_right(theta) = AUC( r(theta) vs r(-theta) )

    flipped if the neuron prefers leftward. This forces p_right(-theta) = 1 - p_right(theta),
    which is the intended symmetry: the antineuron construction cannot express a bias, and
    the neuronal threshold is read off the slope alone.

    Notes
    -----
    Headings are matched to their negatives by value, so the heading set must be symmetric
    about zero (it is: 0, +/-1.5, +/-3, +/-6, +/-12). Zero heading is fixed at 0.5 by
    construction and is retained only to anchor the fit.
    """
    x = np.asarray(headings, float)
    r = np.asarray(rates, float)
    ok = np.isfinite(x) & np.isfinite(r)
    x, r = x[ok], r[ok]

    if prefers_right is None:
        prefers_right = preferred_side(x, r)

    levels = np.unique(x)
    p_right, n_used = [], []
    for h in levels:
        here = r[np.isclose(x, h)]
        there = r[np.isclose(x, -h)]
        if np.isclose(h, 0.0):
            p_right.append(0.5)
            n_used.append(here.size)
            continue
        if here.size < _MIN_TRIALS_PER_GROUP or there.size < _MIN_TRIALS_PER_GROUP:
            p_right.append(np.nan)
            n_used.append(min(here.size, there.size))
            continue
        auc = roc_auc(here, there)
        p_right.append(auc if prefers_right else 1.0 - auc)
        n_used.append(min(here.size, there.size))

    levels = np.asarray(levels, float)
    p_right = np.asarray(p_right, float)
    n_used = np.asarray(n_used, int)

    keep = np.isfinite(p_right)
    if keep.sum() >= 3:
        # Expand the per-heading proportions back into pseudo-trials so the same MLE fitter
        # can be reused, weighting each heading by how many trials supported it.
        xs, ys = [], []
        for h, p, n in zip(levels[keep], p_right[keep], n_used[keep]):
            n_right = int(round(p * n))
            xs.extend([h] * n)
            ys.extend([1.0] * n_right + [0.0] * (n - n_right))
        fit = fit_psychometric(xs, ys, fit_lapse=fit_lapse)
    else:
        fit = PsychometricFit(np.nan, np.nan, np.nan, 0, np.nan, False)

    return NeurometricResult(levels, p_right, n_used, fit, bool(prefers_right))


# --------------------------------------------------------------------------------------
# Choice probability
# --------------------------------------------------------------------------------------
@dataclass
class ChoiceProbability:
    """Grand choice probability and its permutation test."""

    cp: float
    p_value: float
    n_trials: int
    n_conditions: int
    per_heading: dict[float, float] = field(default_factory=dict)


def choice_probability(
    headings: Sequence[float],
    rates: Sequence[float],
    rightward: Sequence[float],
    prefers_right: bool | None = None,
    n_perm: int = 2000,
    min_per_group: int = _MIN_TRIALS_PER_GROUP,
    rng: np.random.Generator | None = None,
) -> ChoiceProbability:
    """Grand choice probability across heading conditions.

    Spike counts are z-scored *within* each heading before pooling, which removes the
    stimulus-driven component of the response so that what remains is trial-to-trial
    covariation with the choice. CP > 0.5 means the unit fires more when the animal reports
    the unit's preferred direction.

    The p-value comes from shuffling choice labels within heading condition, which preserves
    both the per-condition trial counts and the choice imbalance at large headings.
    """
    x = np.asarray(headings, float)
    r = np.asarray(rates, float)
    ch = np.asarray(rightward, float)
    ok = np.isfinite(x) & np.isfinite(r) & np.isfinite(ch)
    x, r, ch = x[ok], r[ok], ch[ok]

    if prefers_right is None:
        prefers_right = preferred_side(x, r)

    z_all, pref_all, cond_all = [], [], []
    per_heading: dict[float, float] = {}

    for i, h in enumerate(np.unique(x)):
        m = np.isclose(x, h)
        rr, cc = r[m], ch[m]
        # A unit that never varies within a condition carries no information about choice.
        if np.sum(cc == 1) < min_per_group or np.sum(cc == 0) < min_per_group:
            continue
        sd = np.std(rr, ddof=1)
        if not np.isfinite(sd) or sd == 0:
            continue
        z = (rr - np.mean(rr)) / sd
        pref = cc if prefers_right else 1.0 - cc
        z_all.append(z)
        pref_all.append(pref)
        cond_all.append(np.full(z.size, i))
        per_heading[float(h)] = roc_auc(z[pref == 1], z[pref == 0])

    if not z_all:
        return ChoiceProbability(np.nan, np.nan, 0, 0, {})

    z = np.concatenate(z_all)
    pref = np.concatenate(pref_all)
    cond = np.concatenate(cond_all)
    cp = roc_auc(z[pref == 1], z[pref == 0])

    p_value = np.nan
    if n_perm > 0 and np.isfinite(cp):
        rng = rng or np.random.default_rng(0)
        null = np.empty(n_perm)
        for k in range(n_perm):
            shuffled = pref.copy()
            for i in np.unique(cond):
                m = cond == i
                shuffled[m] = rng.permutation(shuffled[m])
            null[k] = roc_auc(z[shuffled == 1], z[shuffled == 0])
        # Two-sided, with the observed value included so p is never exactly 0.
        p_value = float(
            (np.sum(np.abs(null - 0.5) >= abs(cp - 0.5)) + 1) / (n_perm + 1)
        )

    return ChoiceProbability(
        float(cp), p_value, int(z.size), len(per_heading), per_heading
    )


# --------------------------------------------------------------------------------------
# Tuning descriptors
# --------------------------------------------------------------------------------------
def discrimination_index(
    conditions: Sequence[float], rates: Sequence[float]
) -> float:
    """Discrimination index (DDI / SSI), as used in Avila 2019 and Takahashi 2007.

    ``DDI = (Rmax - Rmin) / (Rmax - Rmin + 2 * sqrt(SSE / (N - M)))``

    Response modulation relative to intrinsic variability, so unlike a plain modulation index
    it is not inflated by a noisy unit with few trials.
    """
    x = np.asarray(conditions, float)
    r = np.asarray(rates, float)
    ok = np.isfinite(x) & np.isfinite(r)
    x, r = x[ok], r[ok]
    levels = np.unique(x)
    m = levels.size
    n = x.size
    if m < 2 or n <= m:
        return np.nan

    means = np.array([np.mean(r[np.isclose(x, h)]) for h in levels])
    sse = float(np.sum([np.sum((r[np.isclose(x, h)] - mu) ** 2) for h, mu in zip(levels, means)]))
    spread = float(means.max() - means.min())
    denom = spread + 2.0 * np.sqrt(sse / (n - m))
    return spread / denom if denom > 0 else np.nan


def tuning_slope(headings: Sequence[float], rates: Sequence[float]) -> tuple[float, float]:
    """Least-squares slope of rate on heading, and its p-value.

    Over the +/-12 deg range of the discrimination task, heading tuning is close to linear,
    so the slope is the natural summary and its sign defines the preferred direction.
    """
    x = np.asarray(headings, float)
    y = np.asarray(rates, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0:
        return np.nan, np.nan
    res = stats.linregress(x[ok], y[ok])
    return float(res.slope), float(res.pvalue)


def congruency(
    headings_ves: Sequence[float],
    rates_ves: Sequence[float],
    headings_vis: Sequence[float],
    rates_vis: Sequence[float],
    alpha: float = 0.05,
) -> dict[str, float | str]:
    """Classify a unit as congruent or opposite (Gu, Angelaki & DeAngelis 2008).

    Congruent cells prefer the same heading direction for vestibular and visual cues;
    opposite cells prefer mirror directions. Gu 2008's central result is that combining cues
    improves sensitivity for congruent cells and *degrades* it for opposite cells, so this
    classification is a prerequisite for the Fig 3 comparison.

    Congruency index, following the published definition:

        CI = R_ves * R_vis

    where each R is the Pearson correlation coefficient of a linear fit of firing rate against
    heading within that modality. CI runs from -1 to +1; positive means aligned tuning slopes.
    Note this is the product of the two rate-vs-heading correlations, *not* the correlation
    between the two tuning curves -- the two agree in sign but not in magnitude.

    A unit is labelled only when both underlying correlations are individually significant,
    since the sign of a non-significant slope carries no information. Gu 2008 tests whether CI
    itself differs from zero; requiring both components to be significant is a slightly
    stricter and more transparent reading of the same idea.
    """
    s_ves, p_ves = tuning_slope(headings_ves, rates_ves)
    s_vis, p_vis = tuning_slope(headings_vis, rates_vis)

    def _r(h: Sequence[float], r: Sequence[float]) -> float:
        x = np.asarray(h, float)
        y = np.asarray(r, float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.std(y[ok]) == 0:
            return np.nan
        return float(stats.pearsonr(x[ok], y[ok]).statistic)

    r_ves = _r(headings_ves, rates_ves)
    r_vis = _r(headings_vis, rates_vis)
    ci = float(r_ves * r_vis) if np.isfinite(r_ves) and np.isfinite(r_vis) else np.nan

    if not np.isfinite(ci) or p_ves > alpha or p_vis > alpha:
        label = "untuned"
    else:
        label = "congruent" if ci > 0 else "opposite"

    return {
        "slope_ves": s_ves,
        "p_ves": p_ves,
        "slope_vis": s_vis,
        "p_vis": p_vis,
        "r_ves": r_ves,
        "r_vis": r_vis,
        "congruency_index": ci,
        "label": label,
    }


def partial_correlations(
    rates: Sequence[float],
    headings: Sequence[float],
    rightward: Sequence[float],
) -> dict[str, float]:
    """Separate stimulus- from choice-driven activity (Zaidel et al. 2017).

    Choice probability conflates two things: a neuron may covary with the choice because it
    carries the sensory evidence that *drove* the choice, or because it carries a decision
    signal. Partial correlations pull them apart:

    ``r(rate, choice | heading)``  - choice-related activity beyond the stimulus
    ``r(rate, heading | choice)``  - stimulus-related activity beyond the choice

    Zaidel found MSTd heading-dominated and VIP choice-dominated. Where 7a falls is the
    question this project exists to answer.
    """
    r = np.asarray(rates, float)
    h = np.asarray(headings, float)
    c = np.asarray(rightward, float)
    ok = np.isfinite(r) & np.isfinite(h) & np.isfinite(c)
    r, h, c = r[ok], h[ok], c[ok]
    if r.size < 6 or np.std(r) == 0 or np.std(h) == 0 or np.std(c) == 0:
        return {"r_choice_given_heading": np.nan, "r_heading_given_choice": np.nan}

    r_rc = float(stats.pearsonr(r, c).statistic)
    r_rh = float(stats.pearsonr(r, h).statistic)
    r_hc = float(stats.pearsonr(h, c).statistic)

    def _partial(r_xy: float, r_xz: float, r_yz: float) -> float:
        denom = np.sqrt(max((1 - r_xz**2) * (1 - r_yz**2), 1e-12))
        return float((r_xy - r_xz * r_yz) / denom)

    return {
        "r_choice_given_heading": _partial(r_rc, r_rh, r_hc),
        "r_heading_given_choice": _partial(r_rh, r_rc, r_hc),
    }
