"""Psychometric fitting and recovery of the choice sign convention.

The awkward part of this dataset is that the stored ``choice`` field does not straightforwardly
mean "left" or "right". JP's description:

    Pretty sure that in this data, they coded responses as -5 and 5 ... if -5 means 'left' and
    +5 means 'right', then you would expect that just plotting the fraction of +5 responses vs.
    heading should give you a sigmoidal. But pretty sure that if you do that, it gives you a
    gaussian.

A bell shape rather than a sigmoid is the signature of a variable that is coded *relative to the
stimulus* rather than in absolute space -- i.e. something accuracy-like. If the stored value
means "correct", then P(that value) against heading is U-shaped (high at both large headings,
~0.5 at zero); if it means "error", it is the inverted-U that JP remembers seeing. Either way,
absolute direction is recovered by flipping the mapping on one side of zero:

    rightward = (choice == c_hi)  if heading > 0 else (choice == c_lo)

Rather than guess which of the four possible mappings is right, :func:`resolve_choice_coding`
fits all of them and reports why it picked one, so the decision is auditable rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
from scipy import optimize, stats

Scheme = Literal["absolute", "stimulus_relative"]
ZeroPolicy = Literal["exclude", "absolute"]

_EPS = 1e-9


# --------------------------------------------------------------------------------------
# Psychometric function
# --------------------------------------------------------------------------------------
def cumulative_gaussian(
    x: np.ndarray, mu: float, sigma: float, lapse: float = 0.0
) -> np.ndarray:
    """Cumulative Gaussian with an optional symmetric lapse rate.

    ``P(right) = lapse/2 + (1 - lapse) * Phi((x - mu) / sigma)``
    """
    sigma = max(float(sigma), _EPS)
    return lapse / 2.0 + (1.0 - lapse) * stats.norm.cdf((np.asarray(x, float) - mu) / sigma)


@dataclass
class PsychometricFit:
    """Result of a cumulative-Gaussian fit.

    Attributes
    ----------
    mu
        Point of subjective equality (bias), in degrees.
    sigma
        Threshold, in degrees. This is the Gu/Angelaki convention: the standard deviation of
        the fitted cumulative Gaussian, equivalently the 84%-correct point relative to the PSE.
    lapse
        Fitted symmetric lapse rate (0 if lapse fitting was disabled).
    n_trials
        Number of trials entering the fit.
    loglik
        Log-likelihood at the optimum.
    converged
        Whether the optimiser reported success.
    """

    mu: float
    sigma: float
    lapse: float
    n_trials: int
    loglik: float
    converged: bool
    ci: dict[str, tuple[float, float]] = field(default_factory=dict)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return cumulative_gaussian(x, self.mu, self.sigma, self.lapse)


def _neg_loglik(params: np.ndarray, x: np.ndarray, right: np.ndarray, fit_lapse: bool) -> float:
    mu, log_sigma = params[0], params[1]
    lapse = params[2] if fit_lapse else 0.0
    p = cumulative_gaussian(x, mu, np.exp(log_sigma), lapse)
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return -float(np.sum(right * np.log(p) + (1 - right) * np.log(1 - p)))


def fit_psychometric(
    headings: Sequence[float],
    rightward: Sequence[float],
    fit_lapse: bool = True,
    max_lapse: float = 0.2,
    n_boot: int = 0,
    rng: np.random.Generator | None = None,
) -> PsychometricFit:
    """Maximum-likelihood cumulative-Gaussian fit to single-trial choices.

    Parameters
    ----------
    headings
        Heading angle per trial, in degrees. Negative = leftward.
    rightward
        Binary (0/1) rightward choice per trial.
    fit_lapse
        Fit a symmetric lapse rate. Recommended: monkeys break fixation and guess, and an
        unmodelled lapse inflates the threshold estimate.
    n_boot
        If > 0, number of nonparametric bootstrap resamples for confidence intervals.
    """
    x = np.asarray(headings, dtype=float)
    y = np.asarray(rightward, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 4 or len(np.unique(y)) < 2:
        return PsychometricFit(np.nan, np.nan, np.nan, int(x.size), np.nan, False)

    span = max(np.ptp(x), 1.0)
    x0 = [0.0, np.log(span / 4.0)] + ([0.02] if fit_lapse else [])
    bounds = [(-span, span), (np.log(span * 1e-3), np.log(span * 10))]
    if fit_lapse:
        bounds.append((0.0, max_lapse))

    res = optimize.minimize(
        _neg_loglik, x0, args=(x, y, fit_lapse), method="L-BFGS-B", bounds=bounds
    )
    mu = float(res.x[0])
    sigma = float(np.exp(res.x[1]))
    lapse = float(res.x[2]) if fit_lapse else 0.0

    fit = PsychometricFit(mu, sigma, lapse, int(x.size), -float(res.fun), bool(res.success))

    if n_boot > 0:
        rng = rng or np.random.default_rng(0)
        mus, sigmas = [], []
        for _ in range(n_boot):
            idx = rng.integers(0, x.size, x.size)
            b = fit_psychometric(x[idx], y[idx], fit_lapse=fit_lapse, max_lapse=max_lapse)
            if b.converged and np.isfinite(b.sigma):
                mus.append(b.mu)
                sigmas.append(b.sigma)
        if mus:
            fit.ci = {
                "mu": (float(np.percentile(mus, 2.5)), float(np.percentile(mus, 97.5))),
                "sigma": (float(np.percentile(sigmas, 2.5)), float(np.percentile(sigmas, 97.5))),
            }
    return fit


def choice_proportions(
    headings: Sequence[float], rightward: Sequence[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-heading proportion of rightward choices, with trial counts.

    Returns ``(unique_headings, proportion_right, n_trials)``.
    """
    x = np.asarray(headings, dtype=float)
    y = np.asarray(rightward, dtype=float)
    levels = np.unique(x[np.isfinite(x)])
    prop = np.array([np.nanmean(y[x == h]) for h in levels])
    n = np.array([int(np.sum(x == h)) for h in levels])
    return levels, prop, n


# --------------------------------------------------------------------------------------
# Choice coding resolver
# --------------------------------------------------------------------------------------
@dataclass
class ChoiceCoding:
    """How the raw ``choice`` field maps onto a rightward choice.

    Attributes
    ----------
    scheme
        ``"absolute"``  -- the stored value directly encodes left/right.
        ``"stimulus_relative"`` -- the stored value encodes something accuracy-like, so its
        meaning flips with the sign of the heading.
    high_is_right
        Under ``"absolute"``: the higher raw value means rightward.
        Under ``"stimulus_relative"``: the higher raw value on *positive* headings means
        rightward (and therefore means leftward on negative headings).
    values
        The two raw values found in the field.
    diagnostics
        Per-candidate fit quality, and the model-free monotonicity/symmetry scores that
        motivated the choice. Inspect this before trusting the result.
    """

    scheme: Scheme
    high_is_right: bool
    values: tuple[float, float]
    zero_policy: ZeroPolicy
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def apply(
        self, headings: Sequence[float], choice: Sequence[float]
    ) -> np.ndarray:
        """Convert raw choices to a 0/1 rightward indicator (NaN where undefined)."""
        return _apply_scheme(
            headings, choice, self.values, self.scheme, self.high_is_right, self.zero_policy
        )

    def describe(self) -> str:
        lo, hi = self.values
        if self.scheme == "absolute":
            r, l = (hi, lo) if self.high_is_right else (lo, hi)
            return f"absolute coding: {r:g} = rightward, {l:g} = leftward"
        r_pos, r_neg = (hi, lo) if self.high_is_right else (lo, hi)
        return (
            f"stimulus-relative coding: on heading > 0, {r_pos:g} = rightward; "
            f"on heading < 0, {r_neg:g} = rightward "
            f"(zero headings: {self.zero_policy})"
        )


def _apply_scheme(
    headings: Sequence[float],
    choice: Sequence[float],
    values: tuple[float, float],
    scheme: Scheme,
    high_is_right: bool,
    zero_policy: ZeroPolicy,
) -> np.ndarray:
    x = np.asarray(headings, dtype=float)
    c = np.asarray(choice, dtype=float)
    lo, hi = values

    is_hi = np.where(np.isclose(c, hi), 1.0, np.where(np.isclose(c, lo), 0.0, np.nan))
    if not high_is_right:
        is_hi = 1.0 - is_hi

    if scheme == "absolute":
        return is_hi

    # stimulus_relative: meaning flips on negative headings
    right = np.where(x > 0, is_hi, 1.0 - is_hi)
    zero = np.isclose(x, 0.0)
    if zero_policy == "exclude":
        right = np.where(zero, np.nan, right)
    else:  # 'absolute' - fall back to the unflipped mapping at zero heading
        right = np.where(zero, is_hi, right)
    return right


def resolve_choice_coding(
    headings: Sequence[float],
    choice: Sequence[float],
    zero_policy: ZeroPolicy = "exclude",
    fit_lapse: bool = True,
) -> ChoiceCoding:
    """Work out which mapping of ``choice`` onto rightward produces a sigmoid.

    Fits a cumulative Gaussian under each candidate mapping and selects by log-likelihood,
    additionally recording two model-free diagnostics:

    ``monotonicity``
        Spearman rho between per-heading P(right) and heading. A correct mapping gives a
        strongly positive value; an absolute mapping applied to relative data gives ~0.
    ``symmetry``
        Spearman rho between per-heading P(right) and |heading|. Large magnitude means the
        curve is bell- or U-shaped about zero, which is the tell-tale of accuracy coding.

    Raises
    ------
    ValueError
        If ``choice`` does not contain exactly two distinct finite values. Anything else
        (aborts coded in-band, a third "no choice" value) needs handling upstream rather than
        being silently collapsed here.
    """
    x = np.asarray(headings, dtype=float)
    c = np.asarray(choice, dtype=float)
    ok = np.isfinite(x) & np.isfinite(c)
    x, c = x[ok], c[ok]

    uniq = np.unique(c)
    if uniq.size != 2:
        raise ValueError(
            f"expected exactly 2 distinct choice values, found {uniq.size}: {uniq[:10]}. "
            "Filter aborts/no-choice trials before calling resolve_choice_coding()."
        )
    values = (float(uniq[0]), float(uniq[1]))

    candidates: list[dict[str, Any]] = []
    for scheme in ("absolute", "stimulus_relative"):
        for high_is_right in (True, False):
            right = _apply_scheme(x, c, values, scheme, high_is_right, zero_policy)
            keep = np.isfinite(right)
            if keep.sum() < 8:
                continue
            xf, rf = x[keep], right[keep]
            fit = fit_psychometric(xf, rf, fit_lapse=fit_lapse)
            levels, prop, _ = choice_proportions(xf, rf)
            mono = (
                float(stats.spearmanr(levels, prop).statistic) if levels.size > 2 else np.nan
            )
            symm = (
                float(stats.spearmanr(np.abs(levels), prop).statistic)
                if levels.size > 2
                else np.nan
            )
            candidates.append(
                {
                    "scheme": scheme,
                    "high_is_right": high_is_right,
                    "loglik": fit.loglik,
                    "loglik_per_trial": fit.loglik / max(fit.n_trials, 1),
                    "sigma": fit.sigma,
                    "mu": fit.mu,
                    "n_trials": fit.n_trials,
                    "monotonicity": mono,
                    "symmetry": symm,
                    "converged": fit.converged,
                }
            )

    if not candidates:
        raise ValueError("no candidate mapping had enough usable trials")

    # Only mappings with a positive slope are physically sensible; among those take the best
    # per-trial log-likelihood. Per-trial because 'exclude' drops the zero-heading trials and
    # the candidate families would otherwise be compared on different trial counts.
    upright = [k for k in candidates if k["monotonicity"] > 0]
    pool = upright or candidates
    best = max(pool, key=lambda k: k["loglik_per_trial"])

    return ChoiceCoding(
        scheme=best["scheme"],
        high_is_right=best["high_is_right"],
        values=values,
        zero_policy=zero_policy,
        diagnostics={"candidates": candidates, "selected": best},
    )
