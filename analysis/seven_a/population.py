"""Population decoding of heading and choice.

This is the part the template papers did not do, and the part the 7a story probably needs.

Eric's plots show single-unit thresholds of 10-180 deg against a ~3 deg psychophysical
threshold, so no argument resting on single-neuron sensitivity will survive. The productive
version of the question is whether the *population* carries heading well. If it does, and
choice probability is still at chance, then "7a does not feed this decision" is a positive
result rather than a failure to find a signal -- which is exactly the distinction the paper
has to establish.

Two decoders here:

``decode_heading``
    Cross-validated linear discrimination of left vs right from simultaneously recorded
    units, converted to a population neurometric threshold directly comparable to behaviour.

``decode_choice``
    The same, predicting the animal's choice. Run it *within* heading conditions, or on the
    residuals after regressing out heading, so that predicting the choice is not merely
    predicting the stimulus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats

from .behavior import PsychometricFit, fit_psychometric
from .neurometrics import roc_auc


@dataclass
class DecodingResult:
    """Cross-validated decoding performance."""

    accuracy: float
    auc: float
    n_trials: int
    n_units: int
    p_value: float = np.nan
    per_heading: dict[float, float] = None  # type: ignore[assignment]
    fit: PsychometricFit | None = None

    @property
    def threshold(self) -> float:
        """Population neurometric threshold, in degrees, if a fit was performed."""
        return self.fit.sigma if self.fit is not None else np.nan


def _standardise(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd[sd == 0] = 1.0
    return (train - mu) / sd, (test - mu) / sd


def _fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1.0) -> np.ndarray:
    """Ridge-regularised logistic regression by Newton-IRLS.

    Regularisation is not optional here. Population sizes from a 16-channel array are small
    relative to trial counts in some sessions and comparable in others, and at 1-5 spk/s the
    design matrix is close to degenerate; unregularised weights would be dominated by whichever
    unit happened to be silent.
    """
    n, d = x.shape
    xb = np.hstack([x, np.ones((n, 1))])
    w = np.zeros(d + 1)
    penalty = np.eye(d + 1) * l2
    penalty[-1, -1] = 0.0  # never penalise the intercept

    for _ in range(50):
        z = xb @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        grad = xb.T @ (p - y) + penalty @ w
        s = np.clip(p * (1 - p), 1e-6, None)
        hess = (xb * s[:, None]).T @ xb + penalty
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def _predict(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    z = np.hstack([x, np.ones((x.shape[0], 1))]) @ w
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _cv_folds(n: int, k: int, rng: np.random.Generator) -> list[np.ndarray]:
    idx = rng.permutation(n)
    return [f for f in np.array_split(idx, k) if f.size > 0]


def cross_val_decode(
    rates: np.ndarray,
    labels: Sequence[float],
    n_folds: int = 5,
    l2: float = 1.0,
    n_perm: int = 0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float, float]:
    """Cross-validated held-out predictions for a binary label.

    Parameters
    ----------
    rates
        ``(n_trials, n_units)`` firing rates.
    labels
        Binary target, one per trial.

    Returns
    -------
    (held_out_probabilities, auc, permutation_p)
    """
    rng = rng or np.random.default_rng(0)
    x = np.asarray(rates, float)
    y = np.asarray(labels, float)
    ok = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    x, y = x[ok], y[ok]

    n = x.shape[0]
    if n < 2 * n_folds or len(np.unique(y)) < 2:
        return np.full(n, np.nan), np.nan, np.nan

    def _run(target: np.ndarray) -> tuple[np.ndarray, float]:
        pred = np.full(n, np.nan)
        for fold in _cv_folds(n, n_folds, rng):
            train = np.setdiff1d(np.arange(n), fold)
            if len(np.unique(target[train])) < 2:
                continue
            xtr, xte = _standardise(x[train], x[fold])
            w = _fit_logistic(xtr, target[train], l2=l2)
            pred[fold] = _predict(w, xte)
        valid = np.isfinite(pred)
        auc = (
            roc_auc(pred[valid & (target == 1)], pred[valid & (target == 0)])
            if valid.any()
            else np.nan
        )
        return pred, auc

    pred, auc = _run(y)

    p_value = np.nan
    if n_perm > 0 and np.isfinite(auc):
        null = np.array([_run(rng.permutation(y))[1] for _ in range(n_perm)])
        null = null[np.isfinite(null)]
        if null.size:
            p_value = float((np.sum(np.abs(null - 0.5) >= abs(auc - 0.5)) + 1) / (null.size + 1))

    return pred, auc, p_value


def decode_heading(
    rates: np.ndarray,
    headings: Sequence[float],
    n_folds: int = 5,
    l2: float = 1.0,
    n_perm: int = 0,
    rng: np.random.Generator | None = None,
) -> DecodingResult:
    """Decode heading sign from a simultaneously recorded population.

    The decoder is trained on left-vs-right across all non-zero headings, then its held-out
    probability of "rightward" is averaged within each heading to give a **population
    neurometric function**. Fitting that with a cumulative Gaussian yields a threshold in
    degrees that is directly comparable to the psychophysical threshold -- which is the
    comparison that makes the population argument concrete.

    Zero-heading trials are excluded from training (no correct label) but scored.
    """
    rng = rng or np.random.default_rng(0)
    x = np.asarray(rates, float)
    h = np.asarray(headings, float)
    if x.ndim != 2:
        raise ValueError(f"rates must be (n_trials, n_units), got shape {x.shape}")
    if x.shape[0] != h.size:
        raise ValueError(f"rates has {x.shape[0]} trials but headings has {h.size}")

    nonzero = ~np.isclose(h, 0.0) & np.isfinite(h)
    y = (h > 0).astype(float)

    pred, auc, p_value = cross_val_decode(
        x[nonzero], y[nonzero], n_folds=n_folds, l2=l2, n_perm=n_perm, rng=rng
    )

    h_nz = h[nonzero]
    per_heading: dict[float, float] = {}
    for level in np.unique(h_nz):
        m = np.isclose(h_nz, level) & np.isfinite(pred)
        if m.any():
            per_heading[float(level)] = float(np.mean(pred[m] > 0.5))

    fit = None
    if len(per_heading) >= 3:
        xs, ys = [], []
        for level, prop in per_heading.items():
            n_here = int(np.sum(np.isclose(h_nz, level) & np.isfinite(pred)))
            n_right = int(round(prop * n_here))
            xs.extend([level] * n_here)
            ys.extend([1.0] * n_right + [0.0] * (n_here - n_right))
        fit = fit_psychometric(xs, ys, fit_lapse=False)

    valid = np.isfinite(pred)
    accuracy = (
        float(np.mean((pred[valid] > 0.5) == (y[nonzero][valid] == 1))) if valid.any() else np.nan
    )
    return DecodingResult(
        accuracy=accuracy,
        auc=auc,
        n_trials=int(valid.sum()),
        n_units=x.shape[1],
        p_value=p_value,
        per_heading=per_heading,
        fit=fit,
    )


def decode_choice(
    rates: np.ndarray,
    rightward: Sequence[float],
    headings: Sequence[float] | None = None,
    residualise: bool = True,
    n_folds: int = 5,
    l2: float = 1.0,
    n_perm: int = 0,
    rng: np.random.Generator | None = None,
) -> DecodingResult:
    """Decode the animal's choice from population activity.

    Parameters
    ----------
    residualise
        If True (and ``headings`` is given), regress each unit's rate on heading and decode
        from the residuals. Without this, a decoder that "predicts the choice" may only be
        reading the stimulus, since choice and heading are strongly correlated. Residualising
        is the population analogue of z-scoring within heading for choice probability, and it
        is what makes an above-chance result interpretable.
    """
    rng = rng or np.random.default_rng(0)
    x = np.asarray(rates, float).copy()
    y = np.asarray(rightward, float)

    if residualise and headings is not None:
        h = np.asarray(headings, float)
        for level in np.unique(h[np.isfinite(h)]):
            m = np.isclose(h, level)
            if m.sum() > 1:
                x[m] -= x[m].mean(axis=0)

    pred, auc, p_value = cross_val_decode(
        x, y, n_folds=n_folds, l2=l2, n_perm=n_perm, rng=rng
    )
    valid = np.isfinite(pred)
    ok = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    accuracy = (
        float(np.mean((pred[valid] > 0.5) == (y[ok][valid] == 1))) if valid.any() else np.nan
    )
    return DecodingResult(
        accuracy=accuracy,
        auc=auc,
        n_trials=int(valid.sum()),
        n_units=x.shape[1],
        p_value=p_value,
        per_heading={},
    )


def threshold_vs_population_size(
    rates: np.ndarray,
    headings: Sequence[float],
    sizes: Sequence[int] | None = None,
    n_draws: int = 20,
    rng: np.random.Generator | None = None,
) -> dict[int, tuple[float, float]]:
    """Population threshold as a function of how many units are pooled.

    Randomly subsamples units and refits the population neurometric function, which shows
    whether threshold is still falling at the largest population available. If it is, the
    honest statement is that 7a's full population could support the behaviour and our
    recordings simply undersample it -- a claim worth making explicitly rather than leaving
    the reader to wonder.

    Returns ``{n_units: (median_threshold, iqr)}``.
    """
    rng = rng or np.random.default_rng(0)
    x = np.asarray(rates, float)
    n_units = x.shape[1]
    sizes = sizes or [s for s in (1, 2, 4, 8, 16, 32, 64) if s <= n_units]

    out: dict[int, tuple[float, float]] = {}
    for size in sizes:
        thresholds = []
        for _ in range(n_draws):
            cols = rng.choice(n_units, size=size, replace=False)
            res = decode_heading(x[:, cols], headings, rng=rng)
            if np.isfinite(res.threshold):
                thresholds.append(res.threshold)
        if thresholds:
            out[int(size)] = (
                float(np.median(thresholds)),
                float(stats.iqr(thresholds)) if len(thresholds) > 1 else 0.0,
            )
    return out
