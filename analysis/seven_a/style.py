"""Figure style and export helpers.

JP wants panels he can open in Adobe Illustrator, so the non-negotiable part here is that
text stays as text: ``pdf.fonttype = 42`` embeds TrueType rather than converting glyphs to
outlines, and ``svg.fonttype = 'none'`` leaves text as ``<text>`` elements.

Every panel is written as its own file alongside a CSV of the plotted values and a JSON of
the accompanying statistics, so any number that ends up in the manuscript can be traced back
to a file instead of being read off a plot.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Colour convention inherited from the Angelaki-lab figures and Eric's 2018 plots.
# Keeping it identical makes our panels directly comparable to the published ones.
MODALITY_COLORS: dict[str, str] = {
    "ves": "#D62728",  # red    - vestibular
    "vis": "#2CA02C",  # green  - visual
    "com": "#1F77B4",  # blue   - combined
}
MODALITY_LABELS: dict[str, str] = {
    "ves": "Vestibular",
    "vis": "Visual",
    "com": "Combined",
}


def use_illustrator_style() -> None:
    """Apply rcParams that keep exported vector files editable."""
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "lines.linewidth": 1.0,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.transparent": True,
        }
    )


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Mapping):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def save_panel(
    fig: plt.Figure,
    outdir: str | Path,
    name: str,
    data: "object | None" = None,
    stats: Mapping[str, Any] | None = None,
    formats: tuple[str, ...] = ("pdf", "svg"),
) -> Path:
    """Write one panel plus its sidecar data and stats.

    Parameters
    ----------
    fig
        The figure holding a single panel.
    outdir
        Directory to write into; created if absent.
    name
        Stem for the files, e.g. ``"fig1c_threshold_scatter"``.
    data
        Optional ``pandas.DataFrame`` of the plotted values, written as CSV.
    stats
        Optional mapping of statistics, written as JSON.

    Returns
    -------
    Path
        The directory the files were written to.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for ext in formats:
        fig.savefig(outdir / f"{name}.{ext}")

    if data is not None and hasattr(data, "to_csv"):
        data.to_csv(outdir / f"{name}.csv", index=False)

    if stats is not None:
        with open(outdir / f"{name}.stats.json", "w", encoding="utf-8") as fh:
            json.dump(_to_jsonable(stats), fh, indent=2)

    return outdir
