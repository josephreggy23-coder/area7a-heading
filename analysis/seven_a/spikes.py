"""Turning spike times into the firing rates the analyses expect.

Both Gu 2007 and Gu 2008 compute rates over **the middle 1 s of the stimulus presentation**.
Matching that window is not cosmetic: 7a responses are temporally diverse (Avila 2019 report
both unimodal and bimodal time courses, and Eric's peak-response histogram is bimodal for the
vestibular condition), so a window that includes onset and offset transients measures
something different from one that samples the sustained response.

If the dataset ships pre-binned counts rather than spike times, this module is unnecessary --
but then the window is whatever someone chose in 2018, and that should be stated in the paper
rather than assumed.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# Trial structure for the HD task, read off Eric's schematic (image1 of 1D_HD_plots.docx):
# 0.1 s fixation -> 0.4 s static dots -> 2.0 s motion -> 0.4 s static dots.
HD_MOTION_ONSET_S = 0.5
HD_MOTION_DURATION_S = 2.0


def middle_window(
    onset: float = HD_MOTION_ONSET_S,
    duration: float = HD_MOTION_DURATION_S,
    width: float = 1.0,
) -> tuple[float, float]:
    """The middle ``width`` seconds of a stimulus, as ``(start, stop)`` in trial time.

    With the HD defaults this is (1.0, 2.0) s -- the middle 1 s of the 2 s motion period,
    matching Gu 2007 and Gu 2008.
    """
    centre = onset + duration / 2.0
    return centre - width / 2.0, centre + width / 2.0


def rate_in_window(
    spike_times: Sequence[float], start: float, stop: float
) -> float:
    """Mean firing rate (spikes/s) within ``[start, stop)`` for one trial."""
    ts = np.asarray(spike_times, dtype=float)
    ts = ts[np.isfinite(ts)]
    if stop <= start:
        return np.nan
    return float(np.sum((ts >= start) & (ts < stop)) / (stop - start))


def rates_in_window(
    spike_times_per_trial: Sequence[Sequence[float]], start: float, stop: float
) -> np.ndarray:
    """Vectorised :func:`rate_in_window` over trials."""
    return np.array([rate_in_window(ts, start, stop) for ts in spike_times_per_trial])


def psth(
    spike_times_per_trial: Sequence[Sequence[float]],
    t_start: float = 0.0,
    t_stop: float = 2.9,
    bin_width: float = 0.025,
    sigma: float = 0.025,
) -> tuple[np.ndarray, np.ndarray]:
    """Trial-averaged PSTH smoothed with a Gaussian kernel.

    Defaults follow Avila 2019: 25 ms bins, 25 ms Gaussian kernel.

    Returns
    -------
    (bin_centres, rate)
        ``rate`` is in spikes/s.
    """
    edges = np.arange(t_start, t_stop + bin_width, bin_width)
    centres = edges[:-1] + bin_width / 2.0

    counts = np.zeros(centres.size)
    n_trials = 0
    for ts in spike_times_per_trial:
        arr = np.asarray(ts, dtype=float)
        arr = arr[np.isfinite(arr)]
        counts += np.histogram(arr, bins=edges)[0]
        n_trials += 1
    if n_trials == 0:
        return centres, counts

    rate = counts / (n_trials * bin_width)

    if sigma > 0:
        half = int(np.ceil(3 * sigma / bin_width))
        k = np.arange(-half, half + 1) * bin_width
        kernel = np.exp(-0.5 * (k / sigma) ** 2)
        kernel /= kernel.sum()
        rate = np.convolve(rate, kernel, mode="same")

    return centres, rate


def responsiveness(
    spike_times_per_trial: Sequence[Sequence[float]],
    onset: float = HD_MOTION_ONSET_S,
    width: float = 0.25,
) -> tuple[float, float, str]:
    """Is the unit driven by motion? (Avila 2019 criterion.)

    Compares mean rate in the ``width``-second window *after* motion onset against the window
    of the same length immediately *before* it, with a two-sided paired t-test.

    Returns ``(t_statistic, p_value, label)`` where label is ``"excitatory"``,
    ``"suppressive"`` or ``"unresponsive"`` at p <= 0.05.
    """
    from scipy import stats as _stats

    pre = rates_in_window(spike_times_per_trial, onset - width, onset)
    post = rates_in_window(spike_times_per_trial, onset, onset + width)
    ok = np.isfinite(pre) & np.isfinite(post)
    if ok.sum() < 3:
        return np.nan, np.nan, "unresponsive"

    res = _stats.ttest_rel(post[ok], pre[ok])
    t, p = float(res.statistic), float(res.pvalue)
    if not np.isfinite(p) or p > 0.05:
        return t, p, "unresponsive"
    return t, p, "excitatory" if t > 0 else "suppressive"
