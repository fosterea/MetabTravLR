"""Load `metabolite_selection.yaml` (written by `harreman_summary.write_metabolite_selection`)
into the `metabolites` structure `SpatialCellularProgramsEstimator(metabolites=...)` consumes.

The estimator sums each metabolite's transporter gene pairs into ONE modulator column
(`metab@<name>`), so the loader hands it, per metabolite, the list of `(export, import)`
pairs to sum. This loader is responsible for:
  - homotypic pair `(g, g)` -> emit it once (orientation-free);
  - heterotypic pair `{a, b}` -> emit BOTH `(a, b)` and `(b, a)` when `both_orientations=True`
    (directionality is dropped by summing, so both channels feed the same column) -- else
    just the as-given orientation;
  - dedupe pairs *within* a metabolite by unordered identity (a pair listed twice sums once);
  - optionally drop pairs touching a gene absent from `adata.var_names` (drop the metabolite
    entirely if it is left with no pairs);
  - MERGE metabolites whose resulting pair-set is IDENTICAL into a single column (they would
    otherwise be perfectly-collinear duplicate predictors); the merged column's name is the
    joined metabolite names (`nameA|nameB|...`) so it stays searchable.

Note: pairs are NOT deduped *across* metabolites -- a pair shared by several metabolites
contributes to each of their sums (that is the point of a per-metabolite column).

We keep two representations deliberately separate:
  - `load_metabolite_selection` -> `{metabolite: [(g1, g2), ...]}`, the ORIGINAL unordered
    pairs grouped by metabolite (the file, verbatim).
  - `build_metabolites` -> `{column_name: [(export, import), ...]}`, the orientation-expanded,
    var-filtered, merged structure the model wants.
"""
from __future__ import annotations

import yaml

# Separator joining the names of metabolites that merged into one column (identical
# transporter-pair sets). Chosen to not appear in chemical metabolite names (which contain
# commas, spaces, hyphens, parentheses, apostrophes).
MERGE_SEP = "|"


def load_metabolite_selection(path) -> dict[str, list[tuple[str, str]]]:
    """Parse a `metabolite_selection.yaml` into `{metabolite_name: [(g1, g2), ...]}`.

    Pairs are returned as-given (unordered, not deduped, not orientation-expanded).
    Order follows the file (metabolite order, and pair order within each metabolite).
    """
    with open(path) as f:
        doc = yaml.safe_load(f) or {}

    selection: dict[str, list[tuple[str, str]]] = {}
    for entry in doc.get("metabolites", []) or []:
        name = entry["name"]
        pairs = [tuple(pair) for pair in entry.get("gene_pairs", []) or []]
        # If the same metabolite name appears more than once, ACCUMULATE its pairs rather
        # than overwrite (a plain `selection[name] = pairs` would silently drop the earlier
        # entry's transporters). Within-metabolite dedup happens later in `_expand_pairs`.
        if name in selection:
            selection[name] = selection[name] + pairs
        else:
            selection[name] = pairs
    return selection


def _expand_pairs(pairs, var_set, both_orientations):
    """One metabolite's raw pairs -> the (export, import) list to sum.

    Dedupe by unordered identity, orientation-expand heterotypic pairs, and drop pairs
    with a gene absent from `var_set` (when given). Order is deterministic (first-seen).
    """
    seen_unordered: set[frozenset] = set()
    out: list[tuple[str, str]] = []
    for g1, g2 in pairs:
        if var_set is not None and (g1 not in var_set or g2 not in var_set):
            continue
        key = frozenset((g1, g2))
        if key in seen_unordered:
            continue
        seen_unordered.add(key)
        if g1 == g2:
            out.append((g1, g2))
        else:
            out.append((g1, g2))
            if both_orientations:
                out.append((g2, g1))
    return out


def build_metabolites(selection, var_names=None, both_orientations=True) -> dict[str, list[tuple[str, str]]]:
    """Turn `{metabolite: [(g1, g2), ...]}` into `{column_name: [(export, import), ...]}`
    for `SpatialCellularProgramsEstimator(metabolites=...)`.

    See the module docstring for the full contract. Metabolites left empty after
    var-filtering are dropped; metabolites with an identical expanded pair-set are merged
    into one column named by their joined metabolite names (`MERGE_SEP`-separated).

    Order is deterministic (first-seen, following `selection`'s iteration order).
    """
    var_set = set(var_names) if var_names is not None else None

    # 1) expand + var-filter each metabolite; drop empties.
    expanded: dict[str, list[tuple[str, str]]] = {}
    n_dropped_empty = 0
    for name, pairs in selection.items():
        exp = _expand_pairs(pairs, var_set, both_orientations)
        if exp:
            expanded[name] = exp
        else:
            n_dropped_empty += 1

    # 2) merge metabolites with an identical expanded pair-set (order-insensitive).
    order: list[frozenset] = []
    sig_names: dict[frozenset, list[str]] = {}
    sig_pairs: dict[frozenset, list[tuple[str, str]]] = {}
    for name, exp in expanded.items():
        sig = frozenset(exp)
        if sig not in sig_names:
            sig_names[sig] = []
            sig_pairs[sig] = exp
            order.append(sig)
        sig_names[sig].append(name)

    result: dict[str, list[tuple[str, str]]] = {}
    n_merged_groups = 0
    for sig in order:
        names = sig_names[sig]
        if len(names) > 1:
            n_merged_groups += 1
        result[MERGE_SEP.join(names)] = sig_pairs[sig]

    print(
        f"build_metabolites: {len(selection)} metabolites -> {len(result)} columns "
        f"({n_dropped_empty} dropped as empty after var-filter; "
        f"{n_merged_groups} merged group(s) collapsing identical pair-sets)"
    )
    return result


def load_metabolites(path, var_names=None, both_orientations=True):
    """Convenience: parse `path` and return `(metabolites, selection)` in one call --
    `metabolites` for the estimator, `selection` (original grouped pairs) for reference.
    """
    selection = load_metabolite_selection(path)
    metabolites = build_metabolites(selection, var_names=var_names, both_orientations=both_orientations)
    return metabolites, selection
