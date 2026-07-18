"""Load `metabolite_selection.yaml` (written by `harreman_summary.write_metabolite_selection`)
into the exact `metab_pairs` structure `SpatialCellularProgramsEstimator(metab_pairs=...)`
consumes.

The selection file's gene pairs are harreman transporter pairs -- **undirected** (every
transporter is typed `IMP-EXP`, bidirectional) -- and the same unordered pair can appear
under multiple metabolites (many-to-many). There are no scores in this file; significance
filtering already happened when the file was written.

The estimator does NOT add orientations itself -- it builds one betadata column per supplied
`(export, import)` tuple, verbatim. So this loader is responsible for:
  - homotypic pair `(g, g)` -> emit it once (orientation-free);
  - heterotypic pair `{a, b}` -> emit BOTH `(a, b)` and `(b, a)` when `both_orientations=True`
    (D3: both orientations, let group-lasso sort it out) -- else just the as-given orientation;
  - dedupe across all metabolites, so a pair shared by several metabolites yields one column;
  - optionally drop pairs touching a gene absent from `adata.var_names`.

We keep two representations deliberately separate:
  - `load_metabolite_selection` -> `{metabolite: [(g1, g2), ...]}`, the ORIGINAL unordered
    pairs, grouped by metabolite -- needed later to aggregate betas back up to a metabolite.
  - `build_metab_pairs` -> the flat, deduped, both-orientations-expanded list the model wants.
"""
from __future__ import annotations

import yaml


def load_metabolite_selection(path) -> dict[str, list[tuple[str, str]]]:
    """Parse a `metabolite_selection.yaml` into `{metabolite_name: [(g1, g2), ...]}`.

    Pairs are returned as-given (unordered, not deduped, not orientation-expanded) --
    this is the grouping used later to aggregate betas back up to a metabolite. Order
    follows the file (metabolite order, and pair order within each metabolite).
    """
    with open(path) as f:
        doc = yaml.safe_load(f) or {}

    selection: dict[str, list[tuple[str, str]]] = {}
    for entry in doc.get("metabolites", []) or []:
        name = entry["name"]
        pairs = [tuple(pair) for pair in entry.get("gene_pairs", []) or []]
        selection[name] = pairs
    return selection


def build_metab_pairs(selection, var_names=None, both_orientations=True) -> list[tuple[str, str]]:
    """Flatten `{metabolite: [(g1, g2), ...]}` into the deduped `metab_pairs` list the
    estimator consumes.

    - Homotypic pair `(g, g)` -> one tuple `(g, g)`.
    - Heterotypic pair `{a, b}` -> both `(a, b)` and `(b, a)` if `both_orientations`
      (default, D3), else only the as-given orientation.
    - Deduped across metabolites by unordered identity: once an unordered pair has been
      emitted (in either orientation it was first seen), later occurrences under other
      metabolites are skipped.
    - If `var_names` is given (any iterable of gene names), pairs with a gene not present
      are dropped and the drop count is printed.

    Order is deterministic (first-seen, following `selection`'s iteration order) so repeat
    calls on the same input yield an identical list.
    """
    seen_unordered: set[frozenset] = set()
    pairs: list[tuple[str, str]] = []
    for metab_pairs in selection.values():
        for g1, g2 in metab_pairs:
            key = frozenset((g1, g2))
            if key in seen_unordered:
                continue
            seen_unordered.add(key)
            if g1 == g2:
                pairs.append((g1, g2))
            else:
                pairs.append((g1, g2))
                if both_orientations:
                    pairs.append((g2, g1))

    if var_names is not None:
        var_set = set(var_names)
        filtered = [(e, i) for (e, i) in pairs if e in var_set and i in var_set]
        n_dropped = len(pairs) - len(filtered)
        print(f"build_metab_pairs: dropped {n_dropped} of {len(pairs)} metab_pairs "
              f"(gene absent from var_names); kept {len(filtered)}")
        pairs = filtered

    return pairs


def load_metab_pairs(path, var_names=None, both_orientations=True):
    """Convenience: parse `path` and return `(metab_pairs, selection)` in one call --
    `metab_pairs` for the estimator, `selection` for later metabolite-level aggregation.
    """
    selection = load_metabolite_selection(path)
    pairs = build_metab_pairs(selection, var_names=var_names, both_orientations=both_orientations)
    return pairs, selection
