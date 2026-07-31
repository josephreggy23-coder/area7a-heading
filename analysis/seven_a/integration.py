"""Cue-integration predictions and measured weights.

The Bayesian-optimal benchmark for combining two independent cues with Gaussian likelihoods:

    sigma_pred = sqrt(sigma_1^2 * sigma_2^2 / (sigma_1^2 + sigma_2^2))
    w_1        = sigma_2^2 / (sigma_1^2 + sigma_2^2)

Gu 2008 tests the threshold prediction; Gu 2008 and Fetsch 2011 test the weight prediction
using cue-conflict trials. Whether this dataset contains conflict trials is unconfirmed --
see notes/04_data_questions.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import optimize, stats

from .behavior import fit_psychometric


def predicted_threshold(sigma_a: float, sigma_b: float) -> float:
    """Optimal combined threshold from two single-cue thresholds."""
    if not (np.isfinite(sigma_a) and np.isfinite(sigma_b)) or sigma_a <= 0 or sigma_b <= 0:
        return np.nan
    return float(np.sqrt((sigma_a**2 * sigma_b**2) / (sigma_a**2 + sigma_b**2)))


def predicted_weights(sigma_a: float, sigma_b: float) -> tuple[float, float]:
    """Optimal weights ``(w_a, w_b)`` for cues with thresholds ``sigma_a``, ``sigma_b``."""
    if not (np.isfinite(sigma_a) and np.isfinite(sigma_b)) or sigma_a <= 0 or sigma_b <= 0:
        return np.nan, np.nan
    va, vb = sigma_a**2, sigma_b**2
    return float(vb / (va + vb)), float(va / (va + vb))


@dataclass
class IntegrationResult:
    """Measured versus predicted cue combination for one session."""

    sigma_ves: float
    sigma_vis: float
    sigma_com: float
    sigma_pred: float
    w_vis_pred: float
    w_vis_measured: float = np.nan

    @property
    def improvement_ratio(self) -> float:
        """Measured combined threshold divided by the best single cue.

        Below 1 means combining helped; the optimal value is ``sigma_pred / min(single)``,
        which is at most 1/sqrt(2) = 0.707 when the two cues are equally reliable.
        """
        best = np.nanmin([self.sigma_ves, self.sigma_vis])
        return float(self.sigma_com / best) if best > 0 else np.nan

    @property
    def optimality_ratio(self) -> float:
        """Measured combined threshold divided by the optimal prediction.

        1.0 is optimal, > 1 is sub-optimal, < 1 is better than a Bayesian observer with
        independent cues (which usually means correlated noise, or a fitting problem).
        """
        return (
            float(self.sigma_com / self.sigma_pred)
            if np.isfinite(self.sigma_pred) and self.sigma_pred > 0
            else np.nan
        )


def session_integration(
    sigma_ves: float, sigma_vis: float, sigma_com: float, w_vis_measured: float = np.nan
) -> IntegrationResult:
    """Bundle single-cue and combined thresholds with their optimal predictions."""
    w_vis_pred, _ = predicted_weights(sigma_vis, sigma_ves)
    return IntegrationResult(
        sigma_ves=sigma_ves,
        sigma_vis=sigma_vis,
        sigma_com=sigma_com,
        sigma_pred=predicted_threshold(sigma_ves, sigma_vis),
        w_vis_pred=w_vis_pred,
        w_vis_measured=w_vis_measured,
    )


def measured_visual_weight(
    deltas: Sequence[float], pses: Sequence[float]
) -> tuple[float, float]:
    """Estimate the visual weight from PSE shifts across cue-conflict levels.

    Convention (Gu 2008): on a conflict trial with conflict angle ``delta``, the visual cue
    indicates ``delta / 2`` and the vestibular cue ``-delta / 2``. An observer weighting the
    visual cue by ``w_vis`` reports straight-ahead when

        w_vis * (h + delta/2) + (1 - w_vis) * (h - delta/2) = 0
        =>  PSE(delta) = -delta * (w_vis - 1/2)

    so ``w_vis = 1/2 - slope`` where ``slope = d PSE / d delta``.

    Returns ``(w_vis, slope_p_value)``. Needs at least three conflict levels to be meaningful;
    with only two the estimate has no error bar.

    Warning
    -------
    The sign convention for ``delta`` differs between labs and datasets. Verify against a
    session where the two cues are strongly unbalanced before trusting the sign: if the
    recovered weight is anti-correlated with the predicted weight across sessions, the
    convention is flipped.
    """
    d = np.asarray(deltas, float)
    p = np.asarray(pses, float)
    ok = np.isfinite(d) & np.isfinite(p)
    if ok.sum() < 2 or np.ptp(d[ok]) == 0:
        return np.nan, np.nan
    res = stats.linregress(d[ok], p[ok])
    return float(0.5 - res.slope), float(res.pvalue)


def neuronal_weights(
    tuning_ves: Sequence[float],
    tuning_vis: Sequence[float],
    tuning_com: Sequence[float],
    force_nonnegative: bool = False,
) -> dict[str, float]:
    """Estimate how a neuron weights the two cues, from tuning curves alone.

    Gu 2008 fits the combined-condition tuning curve as a linear weighted sum of the two
    single-cue tuning curves,

        R_com = w_ves * R_ves + w_vis * R_vis

    with the weights chosen to minimise the sum-squared error against the measured combined
    response. Weights below 1/2 each indicate the sub-additive combination they report.

    **This does not need cue-conflict trials.** That matters here: even if the dataset turns
    out to have no conflict manipulation -- which would block the *behavioural* weight
    analysis -- the neuronal weight analysis, which is the part that speaks to how 7a itself
    combines cues, remains available.

    Parameters
    ----------
    tuning_ves, tuning_vis, tuning_com
        Mean firing rate per heading, in matching heading order, one entry per heading.
    force_nonnegative
        Constrain both weights to be >= 0. Off by default, since a genuinely negative weight
        is informative (it says the combined response is not a mixture of the two single-cue
        responses at all) and silently clipping it would hide that.

    Returns
    -------
    dict
        ``w_ves``, ``w_vis``, their normalised versions summing to 1, and the ``r_squared`` of
        the fit. A poor ``r_squared`` means the weighted-sum model does not describe this
        neuron and its weights should not be interpreted.
    """
    a = np.asarray(tuning_ves, float)
    b = np.asarray(tuning_vis, float)
    y = np.asarray(tuning_com, float)
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(y)
    if ok.sum() < 3:
        return {"w_ves": np.nan, "w_vis": np.nan, "w_ves_norm": np.nan,
                "w_vis_norm": np.nan, "r_squared": np.nan}

    design = np.column_stack([a[ok], b[ok]])
    if force_nonnegative:
        w, _ = optimize.nnls(design, y[ok])
    else:
        w, *_ = np.linalg.lstsq(design, y[ok], rcond=None)

    pred = design @ w
    ss_res = float(np.sum((y[ok] - pred) ** 2))
    ss_tot = float(np.sum((y[ok] - np.mean(y[ok])) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    total = float(w[0] + w[1])
    return {
        "w_ves": float(w[0]),
        "w_vis": float(w[1]),
        "w_ves_norm": float(w[0] / total) if total != 0 else np.nan,
        "w_vis_norm": float(w[1] / total) if total != 0 else np.nan,
        "r_squared": r2,
    }


def pse_by_conflict(
    headings: Sequence[float],
    rightward: Sequence[float],
    deltas: Sequence[float],
    fit_lapse: bool = True,
) -> dict[float, float]:
    """Fit a psychometric function separately per conflict level and return the PSEs."""
    h = np.asarray(headings, float)
    r = np.asarray(rightward, float)
    d = np.asarray(deltas, float)
    out: dict[float, float] = {}
    for level in np.unique(d[np.isfinite(d)]):
        m = np.isclose(d, level)
        if m.sum() < 8:
            continue
        out[float(level)] = fit_psychometric(h[m], r[m], fit_lapse=fit_lapse).mu
    return out
