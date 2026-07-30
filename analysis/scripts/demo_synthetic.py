"""End-to-end demonstration on synthetic data.

Runs the full pipeline the real recordings will go through and writes Illustrator-ready
panels, so that when the Dropbox data arrives the only thing that changes is the loader.

    python analysis/scripts/demo_synthetic.py --outdir figures/demo

Produces:
    fig1b_psychometric        example session, three modalities
    fig1d_threshold_vs_pred   measured vs optimal combined threshold, all sessions
    fig3a_neurometric         example neurometric functions against behaviour
    fig4b_choice_probability  grand CP distribution per modality
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seven_a import behavior, integration, neurometrics, synth  # noqa: E402
from seven_a.style import (  # noqa: E402
    MODALITY_COLORS,
    MODALITY_LABELS,
    save_panel,
    use_illustrator_style,
)

MODS = ("ves", "vis", "com")


def analyse_session(sess: synth.SynthSession) -> dict:
    """Behaviour + per-unit neurometrics for one session."""
    out: dict = {"psych": {}, "coding": {}, "units": []}

    for mod in MODS:
        h, raw = sess.headings[mod], sess.raw_choice[mod]
        coding = behavior.resolve_choice_coding(h, raw)
        right = coding.apply(h, raw)
        keep = np.isfinite(right)

        out["coding"][mod] = coding
        out["psych"][mod] = behavior.fit_psychometric(h[keep], right[keep])
        out[f"right_{mod}"] = right

        for u in range(sess.rates[mod].shape[0]):
            rates = sess.rates[mod][u]
            nm = neurometrics.neurometric_function(h, rates)
            cp = neurometrics.choice_probability(
                h[keep], rates[keep], right[keep], prefers_right=nm.prefers_right, n_perm=500
            )
            pc = neurometrics.partial_correlations(rates[keep], h[keep], right[keep])
            out["units"].append(
                {
                    "unit": u,
                    "modality": mod,
                    "neuronal_threshold": nm.threshold,
                    "prefers_right": nm.prefers_right,
                    "ddi": neurometrics.discrimination_index(h, rates),
                    "cp": cp.cp,
                    "cp_p": cp.p_value,
                    **pc,
                }
            )

    out["integration"] = integration.session_integration(
        out["psych"]["ves"].sigma, out["psych"]["vis"].sigma, out["psych"]["com"].sigma
    )
    return out


def panel_psychometric(sess, res, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(2.2, 1.9))
    xs = np.linspace(-14, 14, 200)
    rows = []

    for mod in MODS:
        right = res[f"right_{mod}"]
        keep = np.isfinite(right)
        levels, prop, n = behavior.choice_proportions(sess.headings[mod][keep], right[keep])
        fit = res["psych"][mod]
        c = MODALITY_COLORS[mod]

        ax.plot(levels, prop, "o", color=c, ms=3, mfc="none", mew=0.8)
        ax.plot(xs, fit.predict(xs), "-", color=c, lw=1.0,
                label=f"{MODALITY_LABELS[mod]} ({fit.sigma:.2f}$\\degree$)")
        rows += [
            {"modality": mod, "heading": h, "p_right": p, "n": k}
            for h, p, k in zip(levels, prop, n)
        ]

    ax.axhline(0.5, color="0.7", lw=0.5, ls=":")
    ax.set_xlabel("Heading angle (deg)")
    ax.set_ylabel("Proportion rightward")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, loc="upper left", handlelength=1.2)

    save_panel(
        fig, outdir, "fig1b_psychometric",
        data=pd.DataFrame(rows),
        stats={
            mod: {
                "threshold": res["psych"][mod].sigma,
                "pse": res["psych"][mod].mu,
                "lapse": res["psych"][mod].lapse,
                "n_trials": res["psych"][mod].n_trials,
                "choice_coding": res["coding"][mod].describe(),
            }
            for mod in MODS
        },
    )
    plt.close(fig)


def panel_threshold_vs_prediction(results: list[dict], outdir: Path) -> None:
    measured = np.array([r["integration"].sigma_com for r in results])
    predicted = np.array([r["integration"].sigma_pred for r in results])
    ok = np.isfinite(measured) & np.isfinite(predicted)

    fig, ax = plt.subplots(figsize=(1.9, 1.9))
    lim = (0, float(np.nanmax([measured[ok], predicted[ok]])) * 1.15)
    ax.plot(lim, lim, color="0.6", lw=0.6, ls="--")
    ax.plot(predicted[ok], measured[ok], "o", color=MODALITY_COLORS["com"], ms=3.5, alpha=0.8)
    ax.set_xlabel("Predicted threshold (deg)")
    ax.set_ylabel("Measured threshold (deg)")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")

    w = stats.wilcoxon(measured[ok], predicted[ok])
    save_panel(
        fig, outdir, "fig1d_threshold_vs_pred",
        data=pd.DataFrame({"predicted": predicted[ok], "measured": measured[ok]}),
        stats={
            "n_sessions": int(ok.sum()),
            "median_measured": float(np.median(measured[ok])),
            "median_predicted": float(np.median(predicted[ok])),
            "median_optimality_ratio": float(np.median(measured[ok] / predicted[ok])),
            "wilcoxon_statistic": float(w.statistic),
            "wilcoxon_p": float(w.pvalue),
        },
    )
    plt.close(fig)


def panel_neurometric(sess, res, outdir: Path) -> None:
    """Neurometric functions for the most sensitive unit, against behaviour."""
    fig, ax = plt.subplots(figsize=(2.2, 1.9))
    xs = np.linspace(-14, 14, 200)
    rows = []

    units = pd.DataFrame(res["units"])
    for mod in MODS:
        sub = units[(units.modality == mod) & np.isfinite(units.neuronal_threshold)]
        if sub.empty:
            continue
        best = int(sub.loc[sub.neuronal_threshold.idxmin(), "unit"])
        nm = neurometrics.neurometric_function(sess.headings[mod], sess.rates[mod][best])
        c = MODALITY_COLORS[mod]

        ax.plot(nm.headings, nm.p_right, "s", color=c, ms=3, mfc="none", mew=0.8)
        ax.plot(xs, nm.fit.predict(xs), "-", color=c, lw=1.0,
                label=f"{MODALITY_LABELS[mod]} ({nm.threshold:.1f}$\\degree$)")
        ax.plot(xs, res["psych"][mod].predict(xs), ":", color=c, lw=0.8)
        rows += [
            {"modality": mod, "unit": best, "heading": h, "p_right": p}
            for h, p in zip(nm.headings, nm.p_right)
        ]

    ax.axhline(0.5, color="0.7", lw=0.5, ls=":")
    ax.set_xlabel("Heading angle (deg)")
    ax.set_ylabel("Proportion rightward")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, loc="upper left", handlelength=1.2, title="solid: neuron\ndotted: behaviour")
    ax.get_legend().get_title().set_fontsize(6)

    save_panel(fig, outdir, "fig3a_neurometric", data=pd.DataFrame(rows))
    plt.close(fig)


def panel_choice_probability(results: list[dict], outdir: Path) -> None:
    units = pd.concat([pd.DataFrame(r["units"]) for r in results], ignore_index=True)

    fig, ax = plt.subplots(figsize=(2.2, 1.9))
    bins = np.linspace(0.2, 0.8, 25)
    summary = {}

    for mod in MODS:
        cps = units.loc[units.modality == mod, "cp"].to_numpy()
        cps = cps[np.isfinite(cps)]
        if cps.size == 0:
            continue
        ax.hist(cps, bins=bins, histtype="step", color=MODALITY_COLORS[mod],
                lw=1.0, label=MODALITY_LABELS[mod])
        t = stats.wilcoxon(cps - 0.5)
        summary[mod] = {
            "n_units": int(cps.size),
            "mean_cp": float(np.mean(cps)),
            "sem_cp": float(stats.sem(cps)),
            "wilcoxon_p_vs_0.5": float(t.pvalue),
            "n_individually_significant": int(np.sum(units.loc[units.modality == mod, "cp_p"] < 0.05)),
        }

    ax.axvline(0.5, color="0.4", lw=0.7, ls="--")
    ax.set_xlabel("Choice probability")
    ax.set_ylabel("Number of units")
    ax.legend(frameon=False, handlelength=1.2)

    save_panel(fig, outdir, "fig4b_choice_probability", data=units, stats=summary)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", default="figures/demo", type=Path)
    p.add_argument("--n-sessions", default=20, type=int)
    p.add_argument("--seed", default=0, type=int)
    args = p.parse_args()

    use_illustrator_style()
    rng = np.random.default_rng(args.seed)

    sessions, results = [], []
    for _ in range(args.n_sessions):
        sess = synth.simulate_session(
            n_units=10,
            n_reps=30,
            sigma_ves=float(rng.uniform(3.0, 6.0)),
            sigma_vis=float(rng.uniform(2.5, 5.5)),
            rng=rng,
        )
        sessions.append(sess)
        results.append(analyse_session(sess))

    panel_psychometric(sessions[0], results[0], args.outdir)
    panel_neurometric(sessions[0], results[0], args.outdir)
    panel_threshold_vs_prediction(results, args.outdir)
    panel_choice_probability(results, args.outdir)

    ex = results[0]
    print(f"Wrote panels to {args.outdir}\n")
    print("Example session")
    print(f"  choice coding : {ex['coding']['ves'].describe()}")
    for mod in MODS:
        print(f"  {MODALITY_LABELS[mod]:<12} threshold = {ex['psych'][mod].sigma:6.2f} deg")
    print(f"  optimal prediction        = {ex['integration'].sigma_pred:6.2f} deg")
    print(f"  optimality ratio          = {ex['integration'].optimality_ratio:6.2f}")

    units = pd.concat([pd.DataFrame(r["units"]) for r in results], ignore_index=True)
    print(f"\nPopulation ({len(results)} sessions, {len(units)} unit x modality entries)")
    print(f"  median neuronal threshold = {units.neuronal_threshold.median():6.1f} deg")
    print(f"  mean CP                   = {units.cp.mean():6.3f}")
    print(f"  units with p(CP) < 0.05   = {int((units.cp_p < 0.05).sum())} / {len(units)}")


if __name__ == "__main__":
    main()
