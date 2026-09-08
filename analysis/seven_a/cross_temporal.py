"""Cross-temporal generalization analysis for population heading codes.

Standard decoding asks "can we read heading from population activity in the
middle 1 s of the stimulus?" Cross-temporal generalization asks the stronger
question: "if we train a decoder at time t1, how well does it work at time t2?"

The resulting matrix -- accuracy(t_train, t_test) -- reveals the dynamics of
the population code:

- A code that is stable throughout the motion period produces a matrix that is
  uniformly high within the motion epoch. This is what Gu 2008 implicitly
  assumes by averaging over the middle second.

- A code that changes over time (e.g., early sensory representation giving way
  to a late decision signal) produces an off-diagonal drop, meaning the decoder
  trained at one time cannot generalize to another.

- If 7a's heading code is stable (high off-diagonal), the population carries
  the signal long enough to be read out by a downstream decision circuit.
  If it is transient (low off-diagonal), 7a may relay heading but not
  accumulate evidence, which would place it earlier in the decision pipeline
  than MSTd.

Methodology
-----------
The analysis bins spikes into short time windows (default 200 ms), then at
each (t_train, t_test) pair:

1. Train a ridge-logistic classifier on left-vs-right heading using rates
   at t_train.
2. Test it on held-out trials at t_test.
3. Report AUC across cross-validation folds.

The matrix is n_bins x n_bins. Diagonal entries are the standard
time-resolved decoding accuracy; off-diagonal entries are the
generalization measure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .population import _fit_logistic, _predict, _standardise, _cv_folds
from .neurometrics import roc_auc
from .spikes import HD_MOTION_ONSET_S, HD_MOTION_DURATION_S


@dataclass
class CrossTemporalResult:
    """Cross-temporal generalization matrix and metadata."""

    bin_centres: np.ndarray
    matrix: np.ndarray
    n_trials: int
    n_units: int

    @property
    def diagonal(self) -> np.ndarray:
        """Time-resolved decoding (train and test at the same time)."""
        return np.diag(self.matrix)

    @property
    def stability_index(self) -> float:
        """Mean off-diagonal AUC relative to mean diagonal AUC.

        Values near 1.0 indicate a stable code that generalizes across
        time. Values near 0.5/diagonal indicate a code that is specific
        to the training window and does not transfer.

        Computed only over bins within the motion epoch.
        """
        motion_start = HD_MOTION_ONSET_S
        motion_end = HD_MOTION_ONSET_S + HD_MOTION_DURATION_S
        in_motion = (
            (self.bin_centres >= motion_start)
            & (self.bin_centres <= motion_end)
        )
        sub = self.matrix[np.ix_(in_motion, in_motion)]
        if sub.shape[0] < 2:
            return np.nan
        diag = np.diag(sub)
        mask = ~np.eye(sub.shape[0], dtype=bool)
        off_diag = sub[mask]
        mean_diag = float(np.nanmean(diag))
        if mean_diag == 0:
            return np.nan
        return float(np.nanmean(off_diag) / mean_diag)


def bin_spike_trains(
    spike_times_per_trial: Sequence[Sequence[np.ndarray]],
    bin_edges: np.ndarray,
) -> np.ndarray:
    """Bin spike times into firing rates for each unit and time bin.

    Parameters
    ----------
    spike_times_per_trial
        Outer list: trials. Inner list: units. Each element is an array
        of spike times for that (trial, unit) pair.
    bin_edges
        Time bin edges in seconds (n_bins + 1 values).

    Returns
    -------
    rates : ndarray, shape (n_trials, n_units, n_bins)
        Firing rate in spikes/s within each bin.
    """
    n_trials = len(spike_times_per_trial)
    n_units = len(spike_times_per_trial[0])
    n_bins = len(bin_edges) - 1
    bin_width = np.diff(bin_edges)

    rates = np.zeros((n_trials, n_units, n_bins))
    for trial_idx, trial in enumerate(spike_times_per_trial):
        for unit_idx, ts in enumerate(trial):
            ts = np.asarray(ts, dtype=float)
            ts = ts[np.isfinite(ts)]
            counts = np.histogram(ts, bins=bin_edges)[0]
            rates[trial_idx, unit_idx, :] = counts / bin_width

    return rates


def cross_temporal_decoding(
    rates_binned: np.ndarray,
    labels: Sequence[float],
    n_folds: int = 5,
    l2: float = 1.0,
    rng: np.random.Generator | None = None,
) -> CrossTemporalResult:
    """Compute the cross-temporal generalization matrix from pre-binned rates.

    Parameters
    ----------
    rates_binned
        Shape ``(n_trials, n_units, n_bins)`` firing rates per time bin.
    labels
        Binary target per trial (e.g., rightward heading = 1).
    n_folds
        Number of cross-validation folds.
    l2
        Ridge regularization strength for logistic regression.
    rng
        Random generator for fold assignment.

    Returns
    -------
    CrossTemporalResult
        Contains the (n_bins, n_bins) AUC matrix.
    """
    rng = rng or np.random.default_rng(0)
    y = np.asarray(labels, float)
    n_trials, n_units, n_bins = rates_binned.shape

    ok = np.isfinite(y) & np.all(np.all(np.isfinite(rates_binned), axis=1), axis=1)
    rates_ok = rates_binned[ok]
    y_ok = y[ok]
    n = int(ok.sum())

    if n < 2 * n_folds or len(np.unique(y_ok)) < 2:
        return CrossTemporalResult(
            bin_centres=np.array([]),
            matrix=np.full((n_bins, n_bins), np.nan),
            n_trials=n,
            n_units=n_units,
        )

    folds = _cv_folds(n, n_folds, rng)

    # For each training time bin, fit a decoder and store the weights.
    # Then for each test time bin, apply those weights and collect predictions.
    matrix = np.full((n_bins, n_bins), np.nan)

    for t_train in range(n_bins):
        x_train_all = rates_ok[:, :, t_train]

        # Collect held-out predictions for each test time bin.
        preds_per_test = {t: np.full(n, np.nan) for t in range(n_bins)}

        for fold_idx in folds:
            train_idx = np.setdiff1d(np.arange(n), fold_idx)
            if len(np.unique(y_ok[train_idx])) < 2:
                continue

            x_tr, _ = _standardise(x_train_all[train_idx], x_train_all[fold_idx])
            w = _fit_logistic(x_tr, y_ok[train_idx], l2=l2)

            # Apply this decoder to every test time bin.
            for t_test in range(n_bins):
                x_test_all = rates_ok[:, :, t_test]
                _, x_te = _standardise(x_train_all[train_idx], x_test_all[fold_idx])
                preds_per_test[t_test][fold_idx] = _predict(w, x_te)

        # Compute AUC for each test time bin.
        for t_test in range(n_bins):
            pred = preds_per_test[t_test]
            valid = np.isfinite(pred)
            if valid.any():
                pos = pred[valid & (y_ok == 1)]
                neg = pred[valid & (y_ok == 0)]
                if pos.size > 0 and neg.size > 0:
                    matrix[t_train, t_test] = roc_auc(pos, neg)

    return CrossTemporalResult(
        bin_centres=np.array([]),  # caller fills from bin_edges
        matrix=matrix,
        n_trials=n,
        n_units=n_units,
    )


def cross_temporal_heading(
    rates_binned: np.ndarray,
    headings: Sequence[float],
    bin_centres: np.ndarray,
    n_folds: int = 5,
    l2: float = 1.0,
    rng: np.random.Generator | None = None,
) -> CrossTemporalResult:
    """Cross-temporal generalization of heading decoding (left vs right).

    Convenience wrapper that handles label construction from headings and
    attaches bin centres to the result.

    Parameters
    ----------
    rates_binned
        Shape ``(n_trials, n_units, n_bins)``.
    headings
        Heading per trial in degrees. Zero-heading trials are excluded.
    bin_centres
        Centre of each time bin in seconds.
    """
    h = np.asarray(headings, float)
    nonzero = ~np.isclose(h, 0.0) & np.isfinite(h)

    y = (h > 0).astype(float)
    result = cross_temporal_decoding(
        rates_binned[nonzero],
        y[nonzero],
        n_folds=n_folds,
        l2=l2,
        rng=rng,
    )
    result.bin_centres = np.asarray(bin_centres)
    return result


def make_time_bins(
    t_start: float = 0.0,
    t_stop: float = 2.9,
    bin_width: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Create time bin edges and centres for the HD trial epoch.

    Parameters
    ----------
    t_start, t_stop
        Trial time range in seconds.
    bin_width
        Width of each bin in seconds.

    Returns
    -------
    (edges, centres)
        edges has length n_bins + 1, centres has length n_bins.
    """
    edges = np.arange(t_start, t_stop + bin_width / 2, bin_width)
    centres = edges[:-1] + bin_width / 2.0
    return edges, centres
