"""Synthetic data with known ground truth.

The real recordings are not in hand yet, so the pipeline is developed and tested against
simulated sessions whose parameters we set. Two things this buys us:

1. The choice-coding resolver can be validated -- we *encode* choices in the awkward
   stimulus-relative way JP describes and check that the resolver recovers absolute direction.
2. Choice probability, neurometric thresholds and cue-integration predictions can be checked
   against values we know, so a wrong answer on the real data is more likely to be a fact
   about area 7a than a bug in this code.

Firing rates default to the 1-5 spk/s range visible in Eric's HD plots, which is what makes
single-neuron sensitivity in 7a so poor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from scipy import stats

HEADINGS_HD = (-12.0, -6.0, -3.0, -1.5, 0.0, 1.5, 3.0, 6.0, 12.0)
HEADINGS_1D_AZI = tuple(float(a) for a in range(0, 360, 45))


@dataclass
class SynthSession:
    """One simulated session: behaviour plus a population of units."""

    headings: dict[str, np.ndarray]
    rightward: dict[str, np.ndarray]
    raw_choice: dict[str, np.ndarray]
    rates: dict[str, np.ndarray]  # (n_units, n_trials)
    truth: dict[str, float | str]


def encode_choice(
    headings: np.ndarray,
    rightward: np.ndarray,
    scheme: Literal["absolute", "stimulus_relative"] = "stimulus_relative",
    values: tuple[float, float] = (0.0, 5.0),
    high_is_right: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Encode a rightward indicator the way the raw dataset appears to.

    The inverse of :func:`seven_a.behavior.ChoiceCoding.apply`. Under
    ``"stimulus_relative"`` the stored value means "the animal was correct" (or its
    complement), so it flips meaning with the sign of the heading.

    At zero heading there is no correct answer, so a side is assigned at random -- which is
    exactly the ambiguity that makes zero-heading trials unrecoverable under this scheme.
    """
    rng = rng or np.random.default_rng(0)
    lo, hi = values
    right = np.asarray(rightward, float)
    h = np.asarray(headings, float)

    if scheme == "absolute":
        is_hi = right if high_is_right else 1.0 - right
    else:
        flip = np.where(h > 0, 1.0, 0.0)
        zero = np.isclose(h, 0.0)
        flip[zero] = rng.integers(0, 2, int(zero.sum()))
        is_hi = np.where(flip == 1, right, 1.0 - right)
        if not high_is_right:
            is_hi = 1.0 - is_hi

    return np.where(is_hi == 1, hi, lo)


def simulate_session(
    n_units: int = 12,
    n_reps: int = 25,
    headings: Sequence[float] = HEADINGS_HD,
    sigma_ves: float = 4.0,
    sigma_vis: float = 3.5,
    mu: float = 0.0,
    lapse: float = 0.02,
    base_rate: float = 3.0,
    slope_range: tuple[float, float] = (0.02, 0.15),
    frac_congruent: float = 0.6,
    cp_strength: float = 0.35,
    scheme: Literal["absolute", "stimulus_relative"] = "stimulus_relative",
    rng: np.random.Generator | None = None,
) -> SynthSession:
    """Simulate one HD session.

    The combined condition is generated at the Bayesian-optimal threshold, so a correct
    pipeline should recover ``optimality_ratio`` close to 1.

    Parameters
    ----------
    cp_strength
        How strongly a unit's trial-to-trial rate fluctuation is shared with the choice.
        0 gives CP = 0.5; larger values push CP above 0.5 for units preferring rightward.
    frac_congruent
        Fraction of units whose visual tuning has the same sign as their vestibular tuning.
    """
    rng = rng or np.random.default_rng(0)
    h = np.asarray(headings, float)
    sigma_com = float(np.sqrt(sigma_ves**2 * sigma_vis**2 / (sigma_ves**2 + sigma_vis**2)))
    sigmas = {"ves": sigma_ves, "vis": sigma_vis, "com": sigma_com}

    # Each unit gets a vestibular slope; visual slope agrees in sign for congruent units.
    ves_slopes = rng.uniform(*slope_range, n_units) * rng.choice([-1, 1], n_units)
    congruent = rng.random(n_units) < frac_congruent
    vis_slopes = rng.uniform(*slope_range, n_units) * np.where(congruent, 1, -1) * np.sign(
        ves_slopes
    )
    slopes = {"ves": ves_slopes, "vis": vis_slopes, "com": (ves_slopes + vis_slopes) / 2.0}

    out_h, out_right, out_raw, out_rates = {}, {}, {}, {}

    for mod in ("ves", "vis", "com"):
        trial_h = np.repeat(h, n_reps)
        rng.shuffle(trial_h)
        p_right = lapse / 2 + (1 - lapse) * stats.norm.cdf((trial_h - mu) / sigmas[mod])
        right = (rng.random(trial_h.size) < p_right).astype(float)

        # Shared trial-to-trial gain, partially aligned with the choice, is what creates CP.
        shared = rng.normal(0.0, 1.0, trial_h.size)
        choice_signed = np.where(right == 1, 1.0, -1.0)
        latent = shared + cp_strength * choice_signed

        rates = np.empty((n_units, trial_h.size))
        for u in range(n_units):
            lam = base_rate * np.exp(
                slopes[mod][u] * trial_h + 0.25 * latent * np.sign(slopes[mod][u])
            )
            rates[u] = rng.poisson(np.clip(lam, 0.05, None))

        out_h[mod] = trial_h
        out_right[mod] = right
        out_raw[mod] = encode_choice(trial_h, right, scheme=scheme, rng=rng)
        out_rates[mod] = rates

    return SynthSession(
        headings=out_h,
        rightward=out_right,
        raw_choice=out_raw,
        rates=out_rates,
        truth={
            "sigma_ves": sigma_ves,
            "sigma_vis": sigma_vis,
            "sigma_com": sigma_com,
            "mu": mu,
            "lapse": lapse,
            "scheme": scheme,
            "n_congruent": int(congruent.sum()),
            "n_units": n_units,
        },
    )
