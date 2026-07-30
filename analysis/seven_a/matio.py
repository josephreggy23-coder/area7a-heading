"""Loading the MATLAB ``experiments`` structure and flattening it into arrays.

The Dropbox README describes a MATLAB-side load procedure; this module reads the resulting
``.mat`` regardless of whether it was saved as v7 (scipy) or v7.3/HDF5 (h5py), and normalises
it into plain dictionaries of numpy arrays.

The expected shape, from JP's description::

    experiments.singleunits(i).ves.headings
    experiments.singleunits(i).ves.choice
    experiments.singleunits(i).vis.<...>
    experiments.singleunits(i).com.<...>

but the field names are not confirmed, so nothing here hard-codes them beyond a set of
aliases. **Run :func:`describe` first** on the real file and reconcile against
``notes/04_data_questions.md`` before running any analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

MODALITIES = ("ves", "vis", "com")

# Tolerated spellings for the fields we need. Extend as the real file dictates.
FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "headings": ("headings", "heading", "azimuth", "heading_angle", "hdg"),
    "choice": ("choice", "choices", "response", "resp", "decision"),
    "spikes": ("spikes", "spike_counts", "spk", "counts", "spike_rates", "rates", "fr"),
    "spike_times": ("spike_times", "spiketimes", "ts", "spktimes"),
    "coherence": ("coherence", "coh", "motion_coherence"),
    "delta": ("delta", "conflict", "cue_conflict", "displacement"),
    "correct": ("correct", "iscorrect", "outcome", "reward"),
    "trial_num": ("trial_num", "trialnum", "trial", "trial_id"),
}


# --------------------------------------------------------------------------------------
# Raw MATLAB -> python containers
# --------------------------------------------------------------------------------------
def _from_scipy(obj: Any) -> Any:
    """Recursively convert scipy ``mat_struct`` trees into dicts/lists/arrays."""
    import scipy.io as sio

    if isinstance(obj, sio.matlab.mat_struct):
        return {name: _from_scipy(getattr(obj, name)) for name in obj._fieldnames}
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        if obj.ndim == 0:
            return _from_scipy(obj.item())
        return [_from_scipy(x) for x in obj.ravel()]
    return obj


def _from_h5(node: Any, f: Any, depth: int = 0) -> Any:
    """Recursively convert an HDF5 (v7.3 ``.mat``) node, dereferencing object refs."""
    import h5py

    if depth > 32:  # cycles in reference graphs are possible; fail loudly rather than hang
        raise RecursionError("MAT/HDF5 nesting exceeded 32 levels")

    if isinstance(node, h5py.Group):
        return {k: _from_h5(v, f, depth + 1) for k, v in node.items() if not k.startswith("#")}

    if isinstance(node, h5py.Dataset):
        if node.dtype == h5py.ref_dtype:
            refs = np.asarray(node).ravel()
            out = [_from_h5(f[r], f, depth + 1) for r in refs if r]
            return out[0] if len(out) == 1 else out
        data = np.asarray(node)
        # MATLAB char arrays land as uint16 code points.
        if node.attrs.get("MATLAB_class", b"") == b"char":
            return "".join(chr(c) for c in data.ravel().astype(int))
        # HDF5 stores MATLAB matrices transposed.
        return data.T if data.ndim > 1 else data

    return node


def load_mat(path: str | Path) -> dict[str, Any]:
    """Load a ``.mat`` file of either vintage into nested dicts / lists / arrays."""
    path = Path(path)
    try:
        import scipy.io as sio

        raw = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
        return {k: _from_scipy(v) for k, v in raw.items() if not k.startswith("__")}
    except NotImplementedError:
        # v7.3 files are HDF5 and scipy refuses them.
        import h5py

        with h5py.File(path, "r") as f:
            return {k: _from_h5(f[k], f) for k in f.keys() if not k.startswith("#")}


# --------------------------------------------------------------------------------------
# Inspection
# --------------------------------------------------------------------------------------
def describe(obj: Any, name: str = "root", max_depth: int = 6, max_items: int = 12) -> str:
    """Render the structure as an indented tree.

    This is the first thing to run on the real file. It prints field names, container sizes,
    array shapes and dtypes, and for small numeric arrays the unique values -- which is how
    you find out what the ``choice`` field actually contains.
    """
    lines: list[str] = []

    def walk(node: Any, label: str, depth: int) -> None:
        pad = "  " * depth
        if depth > max_depth:
            lines.append(f"{pad}{label}: ...")
            return

        if isinstance(node, dict):
            lines.append(f"{pad}{label}: struct ({len(node)} fields)")
            for k, v in list(node.items())[:max_items]:
                walk(v, k, depth + 1)
            if len(node) > max_items:
                lines.append(f"{pad}  ... {len(node) - max_items} more fields")

        elif isinstance(node, (list, tuple)):
            lines.append(f"{pad}{label}: array of {len(node)} elements")
            if node:
                walk(node[0], f"{label}[0]", depth + 1)

        elif isinstance(node, np.ndarray):
            desc = f"{pad}{label}: ndarray {node.shape} {node.dtype}"
            flat = node.ravel()
            if node.size and np.issubdtype(node.dtype, np.number):
                uniq = np.unique(flat[np.isfinite(flat)]) if flat.size < 500_000 else None
                if uniq is not None and uniq.size <= 12:
                    desc += f"  unique={np.round(uniq, 4).tolist()}"
                elif uniq is not None and uniq.size:
                    desc += f"  range=[{uniq.min():.4g}, {uniq.max():.4g}]"
            lines.append(desc)

        elif isinstance(node, str):
            lines.append(f"{pad}{label}: str {node[:60]!r}")
        else:
            lines.append(f"{pad}{label}: {type(node).__name__} {node!r}"[:120])

    walk(obj, name, 0)
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------------------
def find_field(struct: Any, key: str, aliases: dict[str, tuple[str, ...]] | None = None) -> Any:
    """Fetch ``key`` from a struct, tolerating alternative spellings and case.

    Returns ``None`` if nothing matches, so callers can distinguish "field absent" from
    "field present but empty" -- a distinction that matters for coherence and conflict, whose
    absence changes which analyses are possible at all.
    """
    if not isinstance(struct, dict):
        return None
    aliases = aliases or FIELD_ALIASES
    candidates = aliases.get(key, (key,))
    lowered = {k.lower().replace("_", ""): k for k in struct}
    for cand in candidates:
        hit = lowered.get(cand.lower().replace("_", ""))
        if hit is not None:
            return struct[hit]
    return None


def _as_1d(x: Any) -> np.ndarray:
    if x is None:
        return np.array([])
    arr = np.asarray(x)
    if arr.dtype == object:
        return arr
    return arr.ravel()


@dataclass
class UnitData:
    """One recorded unit, with per-modality trial arrays.

    ``by_modality`` maps ``"ves"`` / ``"vis"`` / ``"com"`` to a dict of 1-D trial arrays
    (``headings``, ``choice``, ``spikes``, and whatever else was present).
    """

    index: int
    by_modality: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def modality(self, name: str) -> dict[str, np.ndarray]:
        return self.by_modality.get(name, {})

    def has(self, name: str, key: str) -> bool:
        m = self.by_modality.get(name, {})
        return key in m and m[key].size > 0

    @property
    def modalities(self) -> tuple[str, ...]:
        return tuple(k for k in MODALITIES if k in self.by_modality)


def extract_units(
    data: dict[str, Any],
    root: str = "experiments",
    collection: str = "singleunits",
    extra_fields: Sequence[str] = (),
) -> list[UnitData]:
    """Flatten ``experiments.singleunits`` into a list of :class:`UnitData`.

    Pulls the canonical fields (``headings``, ``choice``, ``spikes``) plus ``coherence``,
    ``delta``, ``correct`` and ``trial_num`` when present, plus anything named in
    ``extra_fields``.
    """
    node = data.get(root, data)
    if isinstance(node, dict):
        units_node = find_field(node, collection, {collection: (collection, "singleunit", "su")})
        if units_node is None:
            units_node = node.get(collection)
    else:
        units_node = node

    if units_node is None:
        raise KeyError(
            f"could not find '{root}.{collection}'. Run describe() on the file and pass the "
            "correct root/collection names."
        )

    if isinstance(units_node, dict):
        units_node = [units_node]

    wanted = ("headings", "choice", "spikes", "spike_times", "coherence", "delta",
              "correct", "trial_num", *extra_fields)

    units: list[UnitData] = []
    for i, raw_unit in enumerate(units_node):
        if not isinstance(raw_unit, dict):
            continue
        unit = UnitData(index=i)
        for mod in MODALITIES:
            sub = find_field(raw_unit, mod, {mod: (mod,)})
            if not isinstance(sub, dict):
                continue
            trials: dict[str, np.ndarray] = {}
            for key in wanted:
                val = find_field(sub, key)
                if val is not None:
                    arr = _as_1d(val)
                    if arr.size:
                        trials[key] = arr
            if trials:
                unit.by_modality[mod] = trials
        unit.meta = {
            k: v for k, v in raw_unit.items()
            if k.lower() not in MODALITIES and not isinstance(v, (dict, list))
        }
        units.append(unit)
    return units


def iter_modality_trials(units: Sequence[UnitData]) -> Iterator[tuple[UnitData, str, dict]]:
    """Yield ``(unit, modality, trials)`` for every unit x modality present."""
    for unit in units:
        for mod in unit.modalities:
            yield unit, mod, unit.by_modality[mod]
