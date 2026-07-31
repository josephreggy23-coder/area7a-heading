"""Run this FIRST on the real .mat file, before any analysis.

    python analysis/scripts/inspect_data.py data/<file>.mat

Prints the structure tree, then answers the specific questions in
notes/04_data_questions.md -- what the choice field contains, whether cue-conflict or
coherence fields exist, how many trials per condition, and what the firing rates look like.

Nothing here assumes any field name is correct. If the report says a field is missing, that
is information, not an error: the two "missing" cases (conflict, coherence) each rule out a
specific analysis, and it is better to learn that in ten seconds than after a week.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seven_a import behavior  # noqa: E402
from seven_a.matio import MODALITIES, describe, extract_units, load_mat  # noqa: E402


def _report_choice_field(units, mod: str = "ves") -> None:
    print(f"\n{'=' * 78}\nCHOICE CODING ({mod})\n{'=' * 78}")

    headings, choices = [], []
    for u in units:
        t = u.modality(mod)
        if "headings" in t and "choice" in t and t["headings"].size == t["choice"].size:
            headings.append(t["headings"])
            choices.append(t["choice"])
    if not headings:
        print("  no unit had matching headings and choice arrays -- check field names above")
        return

    h = np.concatenate(headings).astype(float)
    c = np.concatenate(choices).astype(float)
    print(f"  pooled trials      : {h.size}")
    print(f"  heading values     : {np.unique(h[np.isfinite(h)]).tolist()}")

    uniq = np.unique(c[np.isfinite(c)])
    print(f"  choice values      : {uniq.tolist()}")
    if uniq.size != 2:
        print(f"  !! expected 2 distinct choice values, found {uniq.size}.")
        print("     Filter aborts / no-choice trials before fitting. This is question 3 in")
        print("     notes/04_data_questions.md.")
        return

    try:
        coding = behavior.resolve_choice_coding(h, c)
    except ValueError as exc:
        print(f"  !! resolver failed: {exc}")
        return

    print(f"\n  RESOLVED -> {coding.describe()}")
    print("\n  candidates (monotonicity should be ~0 for the wrong scheme,")
    print("              |symmetry| large means a bell shape i.e. accuracy coding):")
    print(f"    {'scheme':<20} {'hi=right':<9} {'LL/trial':>9} {'sigma':>8} "
          f"{'mono':>7} {'symm':>7}")
    for cand in coding.diagnostics["candidates"]:
        mark = "  <-" if cand is coding.diagnostics["selected"] else ""
        print(f"    {cand['scheme']:<20} {str(cand['high_is_right']):<9} "
              f"{cand['loglik_per_trial']:>9.4f} {cand['sigma']:>8.2f} "
              f"{cand['monotonicity']:>7.3f} {cand['symmetry']:>7.3f}{mark}")

    right = coding.apply(h, c)
    levels, prop, n = behavior.choice_proportions(h[np.isfinite(right)], right[np.isfinite(right)])
    print("\n  resulting psychometric (should be monotonically increasing):")
    for lv, p, k in zip(levels, prop, n):
        bar = "#" * int(round(p * 40))
        print(f"    {lv:>7.1f} deg  {p:5.3f}  (n={k:>4})  {bar}")

    if coding.scheme == "stimulus_relative":
        n_zero = int(np.sum(np.isclose(h, 0.0)))
        print(f"\n  NOTE: {n_zero} zero-heading trials are undecodable under this scheme.")
        print("  Those are the most informative trials for choice probability. Look for a")
        print("  separate field recording the rewarded side or chosen target.")


def _report_optional_fields(units) -> None:
    print(f"\n{'=' * 78}\nGATING FIELDS\n{'=' * 78}")

    for key, blocks in (
        ("delta", "behavioural cue weights from PSE shifts (Gu 2008)"),
        ("coherence", "the entire Fetsch 2011 reliability replication"),
    ):
        found = {}
        for u in units:
            for mod in u.modalities:
                arr = u.modality(mod).get(key)
                if arr is not None and arr.size:
                    vals = np.unique(arr[np.isfinite(arr.astype(float))])
                    found.setdefault(mod, set()).update(vals.tolist())
        if found:
            print(f"  {key:<10} FOUND: " + ", ".join(
                f"{m}={sorted(v)}" for m, v in sorted(found.items())))
            if all(len(v) < 2 for v in found.values()):
                print(f"             but only one distinct value -- still blocks {blocks}")
        else:
            print(f"  {key:<10} ABSENT -> blocks {blocks}")

    for key in ("spike_times", "spikes", "correct", "trial_num"):
        present = sum(1 for u in units for m in u.modalities if u.has(m, key))
        print(f"  {key:<10} present in {present} unit x modality entries")


def _report_counts_and_rates(units) -> None:
    print(f"\n{'=' * 78}\nTRIAL COUNTS AND FIRING RATES\n{'=' * 78}")
    print(f"  units: {len(units)}")

    for mod in MODALITIES:
        n_units = sum(1 for u in units if mod in u.by_modality)
        if not n_units:
            continue
        per_trial, rates = [], []
        for u in units:
            t = u.modality(mod)
            if "headings" in t:
                h = t["headings"].astype(float)
                levels, counts = np.unique(h[np.isfinite(h)], return_counts=True)
                per_trial.append(counts.min() if counts.size else 0)
            arr = t.get("spikes")
            if arr is not None and arr.size and np.issubdtype(arr.dtype, np.number):
                rates.append(float(np.nanmean(arr.astype(float))))

        line = f"  {mod:<4} units={n_units:<4}"
        if per_trial:
            line += f" min trials/heading: median={np.median(per_trial):.0f}"
            thin = int(np.sum(np.array(per_trial) < 6))
            line += f", {thin} unit(s) below 6"
        if rates:
            line += f" | mean rate={np.mean(rates):.2f} spk/s"
        print(line)

    print("\n  Choice probability needs >=3 trials of EACH choice per heading (Gu 2007).")
    print("  Units below that threshold contribute nothing and should be excluded, not fit.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path, help="path to the .mat file")
    p.add_argument("--root", default="experiments")
    p.add_argument("--collection", default="singleunits")
    p.add_argument("--depth", default=7, type=int)
    p.add_argument("--tree-only", action="store_true")
    args = p.parse_args()

    if not args.path.exists():
        sys.exit(f"not found: {args.path}\nDownload the HD file from the 'VisVest Eric' "
                 "Dropbox folder into data/ first.")

    print(f"Loading {args.path} ({args.path.stat().st_size / 1e6:.1f} MB)...")
    data = load_mat(args.path)

    print(f"\n{'=' * 78}\nSTRUCTURE\n{'=' * 78}")
    print(describe(data, max_depth=args.depth))
    if args.tree_only:
        return

    try:
        units = extract_units(data, root=args.root, collection=args.collection)
    except KeyError as exc:
        sys.exit(f"\n{exc}\n\nUse the tree above to find the right --root / --collection.")

    print(f"\nExtracted {len(units)} units.")
    if not units:
        sys.exit("No units extracted -- check the tree above against matio.FIELD_ALIASES.")

    print(f"Modalities present on unit 0: {units[0].modalities}")
    _report_counts_and_rates(units)
    _report_optional_fields(units)
    for mod in units[0].modalities:
        _report_choice_field(units, mod)


if __name__ == "__main__":
    main()
