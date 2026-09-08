"""Tests for cross-temporal generalization analysis."""

from __future__ import annotations

import numpy as np
import pytest

from seven_a import cross_temporal, synth


def _make_binned_population(
    n_units: int,
    n_reps: int,
    n_bins: int,
    slope: float,
    rng: np.random.Generator,
    stable: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Create simulated binned rates with known temporal structure.

    If stable=True, the heading signal is the same in every time bin
    (a stable code). If stable=False, the signal flips sign halfway
    through (an unstable code that should not generalize).
    """
    headings = np.repeat(synth.HEADINGS_HD, n_reps)
    n_trials = headings.size
    rates = np.empty((n_trials, n_units, n_bins))

    for b in range(n_bins):
        effective_slope = slope if (stable or b < n_bins // 2) else -slope
        for u in range(n_units):
            sign = rng.choice([-1, 1])
            lam = np.clip(3.0 * np.exp(effective_slope * headings * sign), 0.05, None)
            rates[:, u, b] = rng.poisson(lam)

    return rates, headings


def test_stable_code_has_high_off_diagonal():
    """A population with a stable heading code should generalize across time."""
    rng = np.random.default_rng(0)
    n_bins = 6
    rates, headings = _make_binned_population(
        n_units=16, n_reps=60, n_bins=n_bins, slope=0.10, rng=rng, stable=True
    )
    _, centres = cross_temporal.make_time_bins(0.0, 2.4, 0.4)

    result = cross_temporal.cross_temporal_heading(
        rates, headings, centres, rng=rng
    )

    assert result.matrix.shape == (n_bins, n_bins)
    # Diagonal should be above chance.
    diag = result.diagonal
    valid_diag = diag[np.isfinite(diag)]
    assert np.mean(valid_diag) > 0.6

    # Off-diagonal should also be above chance for a stable code.
    mask = ~np.eye(n_bins, dtype=bool)
    off_diag = result.matrix[mask]
    valid_off = off_diag[np.isfinite(off_diag)]
    assert np.mean(valid_off) > 0.55


def test_unstable_code_has_low_cross_temporal_generalization():
    """When the code flips halfway, early-trained decoders should fail on late bins."""
    rng = np.random.default_rng(1)
    n_bins = 6
    rates, headings = _make_binned_population(
        n_units=16, n_reps=80, n_bins=n_bins, slope=0.12, rng=rng, stable=False
    )
    _, centres = cross_temporal.make_time_bins(0.0, 2.4, 0.4)

    result = cross_temporal.cross_temporal_heading(
        rates, headings, centres, rng=rng
    )

    # Cross-block (early train, late test) should be below diagonal.
    early = slice(0, n_bins // 2)
    late = slice(n_bins // 2, n_bins)
    cross_block = result.matrix[early, late]
    valid_cross = cross_block[np.isfinite(cross_block)]
    # When the code flips, cross-block AUC should be near or below 0.5.
    assert np.mean(valid_cross) < 0.55


def test_stability_index_near_one_for_stable_code():
    rng = np.random.default_rng(2)
    rates, headings = _make_binned_population(
        n_units=16, n_reps=60, n_bins=6, slope=0.10, rng=rng, stable=True
    )
    edges, centres = cross_temporal.make_time_bins(0.5, 2.5, 1.0 / 3)

    result = cross_temporal.cross_temporal_heading(
        rates, headings, centres, rng=rng
    )

    assert np.isfinite(result.stability_index)
    assert result.stability_index > 0.5


def test_make_time_bins_covers_trial():
    edges, centres = cross_temporal.make_time_bins(0.0, 2.9, 0.2)

    assert edges[0] == 0.0
    # Last edge should be at or near t_stop (within one bin width).
    assert edges[-1] >= 2.9 - 0.2
    assert centres.size == edges.size - 1
    assert np.all(np.diff(edges) == pytest.approx(0.2))


def test_result_metadata():
    rng = np.random.default_rng(3)
    rates, headings = _make_binned_population(
        n_units=8, n_reps=40, n_bins=4, slope=0.08, rng=rng, stable=True
    )
    _, centres = cross_temporal.make_time_bins(0.0, 1.6, 0.4)

    result = cross_temporal.cross_temporal_heading(
        rates, headings, centres, rng=rng
    )

    assert result.n_units == 8
    assert result.n_trials > 0
    assert result.matrix.shape == (4, 4)
    assert result.bin_centres.size == 4
