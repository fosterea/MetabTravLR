"""Read trained betas back out of `betadata/` and summarize them per tier.

A tier is an `adata.obs` column of cell-type labels (e.g. `Tier1`). For each tier we
group cells by their label and take the mean/std of each modulator's beta.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SEPARATORS = {"@": "metab", "$": "lr", "#": "ltf"}

METAB_PREFIX = "metab@"


def _group(modulator):
    """Modulator group: 'metab', 'lr', 'ltf', or 'tf' (no separator).

    A metabolite column is named 'metab@<name>' (the '@' keys it to the metab group);
    L-R uses '$', L-TF uses '#'; a bare gene name (no separator) is a TF.
    """
    for sep, name in SEPARATORS.items():
        if sep in modulator:
            return name
    return "tf"


def _read_betas(path, group=None):
    """One gene's betadata as cells x modulators, columns stripped of the 'beta_' prefix."""
    cols = [c for c in pq.read_schema(path).names if c.startswith("beta_")]
    names = [c[len("beta_"):] for c in cols]
    if group is not None:
        cols = [c for c, n in zip(cols, names) if _group(n) == group]
        names = [n for n in names if _group(n) == group]
    betas = pd.read_parquet(path, columns=cols)
    betas.columns = names
    return betas


def _gene_paths(betadata_dir, genes=None):
    paths = sorted(Path(betadata_dir).glob("*_betadata.parquet"))
    found = [(p.name[: -len("_betadata.parquet")], p) for p in paths]
    if genes is None:
        return found
    return [(g, p) for g, p in found if g in set(genes)]


def tier_means(betadata_dir, obs, tier, genes=None, group=None):
    """Mean/std/n of every beta per (gene, modulator, cell type), cell type = `obs[tier]`.

    `group` restricts to one modulator group; `genes` restricts to those target genes.
    """
    frames = []
    for gene, path in _gene_paths(betadata_dir, genes):
        betas = _read_betas(path, group)
        if betas.empty or betas.columns.empty:
            continue
        cell_types = obs.loc[betas.index, tier]
        grouped = betas.groupby(cell_types.values, observed=True)
        stats = pd.concat(
            {"mean": grouped.mean().stack(), "std": grouped.std().stack(), "n": grouped.count().stack()},
            axis=1,
        )
        stats.index.names = ["cell_type", "modulator"]
        stats = stats.reset_index()
        stats.insert(0, "gene", gene)
        frames.append(stats)

    if not frames:
        return pd.DataFrame(columns=["gene", "cell_type", "modulator", "mean", "std", "n"])
    return pd.concat(frames, ignore_index=True)


# Per modulator group: (filename, separator, split column names). A metabolite is ONE summed
# column, so it is written as a single `metabolite` column (the 'metab@' marker stripped; a
# merged column keeps its 'nameA|nameB' joined name). L-R/L-TF split their modulator into two
# columns plus a full-string `pair`; TF keeps a single bare `tf` column.
_GROUP_OUTPUTS = {
    "metab": ("metabolites.csv", None, ["metabolite"]),
    "lr": ("ligand_receptor.csv", "$", ["ligand", "receptor"]),
    "ltf": ("ligand_tf.csv", "#", ["ligand", "tf"]),
    "tf": ("transcription_factor.csv", None, ["tf"]),
}


def _write_group(betadata_dir, obs, tier, genes, tier_dir, group):
    """Write one modulator group's per-(gene, modulator, cell type) betas to its CSV.

    Metab -> a single `metabolite` column (the 'metab@' marker stripped). L–R/L–TF split their
    modulator into two columns plus a full-string `pair`; TF keeps a single `tf` column.
    Returns the written DataFrame.
    """
    filename, sep, parts = _GROUP_OUTPUTS[group]
    stats = tier_means(betadata_dir, obs, tier, genes, group=group)

    if group == "metab":  # single summed column; strip the 'metab@' marker
        col = parts[0]
        if stats.empty:
            stats[col] = pd.Series(dtype=object)
        else:
            stats[col] = stats["modulator"].str.replace(f"^{METAB_PREFIX}", "", regex=True)
        stats = stats[["gene", col, "cell_type", "mean", "std", "n"]]
    elif sep is None:  # TF: modulator is the bare gene name
        stats = stats.rename(columns={"modulator": parts[0]})
        stats = stats[["gene", parts[0], "cell_type", "mean", "std", "n"]]
    else:
        if stats.empty:  # nothing to split; keep the columns so the CSV header is stable
            for p in parts:
                stats[p] = pd.Series(dtype=object)
        else:
            stats[parts] = stats["modulator"].str.split(sep, n=1, expand=True)
        stats = stats.rename(columns={"modulator": "pair"})
        stats = stats[["gene", *parts, "pair", "cell_type", "mean", "std", "n"]]

    stats.to_csv(tier_dir / filename, index=False)
    return stats


def write_metabolites(betadata_dir, obs, tiers, outdir, genes=None, write_all_groups=True):
    """Write `<outdir>/<tier>/metabolites.csv`: metabolite betas per (gene, metabolite, cell
    type). Each metabolite is one summed column (`metabolite` = the name, 'metab@' stripped;
    merged identical-signature metabolites keep their 'nameA|nameB' joined name). Returns
    {tier: DataFrame} (the metabolite frame per tier).

    When `write_all_groups` is True (default), also write the other learned-beta groups next to
    it, one CSV per tier: `ligand_receptor.csv` (ligand/receptor cols), `ligand_tf.csv`
    (ligand/tf cols), and `transcription_factor.csv` (tf col) — so all coefficients are saved
    for later viz/analysis.
    """
    out = {}
    for tier in tiers:
        tier_dir = Path(outdir) / tier
        tier_dir.mkdir(parents=True, exist_ok=True)

        out[tier] = _write_group(betadata_dir, obs, tier, genes, tier_dir, "metab")
        if write_all_groups:
            for group in ("lr", "ltf", "tf"):
                _write_group(betadata_dir, obs, tier, genes, tier_dir, group)
    return out


def write_histograms(betadata_dir, obs, tiers, outdir, genes=None, bins=50, plot=False):
    """Write `<outdir>/<tier>/histograms.csv`: distribution of the per-(gene, modulator,
    cell type) mean betas, one histogram per modulator group. Returns {tier: DataFrame}.
    """
    out = {}
    for tier in tiers:
        means = tier_means(betadata_dir, obs, tier, genes)
        means["group"] = [_group(m) for m in means["modulator"]]

        rows = []
        for group, sub in means.groupby("group"):
            counts, edges = np.histogram(sub["mean"].dropna(), bins=bins)
            rows.append(pd.DataFrame({
                "group": group,
                "left": edges[:-1],
                "right": edges[1:],
                "count": counts,
            }))
        hist = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
            columns=["group", "left", "right", "count"])

        tier_dir = Path(outdir) / tier
        tier_dir.mkdir(parents=True, exist_ok=True)
        hist.to_csv(tier_dir / "histograms.csv", index=False)
        if plot:
            _plot_histograms(hist, tier_dir / "histograms.png", tier)
        out[tier] = hist
    return out


def _plot_histograms(hist, path, title):
    import matplotlib.pyplot as plt

    groups = list(hist["group"].unique())
    fig, axes = plt.subplots(1, len(groups), figsize=(4 * len(groups), 3), squeeze=False)
    for ax, group in zip(axes[0], groups):
        sub = hist[hist["group"] == group]
        ax.bar(sub["left"], sub["count"], width=sub["right"] - sub["left"], align="edge")
        ax.set_title(group)
        ax.set_xlabel("mean beta")
    axes[0][0].set_ylabel("count")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def betas_to_adata(adata, betadata_dir, genes=None, group="metab"):
    """Attach per-cell betas to `adata.obsm['beta_<gene>']` (cells x modulators).

    Cells the gene was not fit on get NaN. Modulator names go in
    `adata.uns['beta_modulators'][gene]`, since obsm drops column labels.
    """
    modulators = adata.uns.setdefault("beta_modulators", {})
    for gene, path in _gene_paths(betadata_dir, genes):
        betas = _read_betas(path, group).reindex(adata.obs_names)
        adata.obsm[f"beta_{gene}"] = betas.to_numpy()
        modulators[gene] = list(betas.columns)
    return adata


def compute_metab_x(adata, metabolites, radius, contact_distance=50,
                    scale_factor=100, layer="imputed_count"):
    """The metabolite *communication-score* matrix (cells x metabolites) that the metab betas
    multiply -- i.e. the `x` in `y = ... + beta_metab@<name> * x`.

    For metabolite `name` with transporter pairs [(e1,i1), (e2,i2), ...], the model's design
    matrix column `metab@<name>` is `sum_k received(e_k) * import_raw(i_k)`, where `received(.)`
    is the Gaussian-distance-weighted (diffused) export expression and `import_raw(.)` is the raw
    import expression in `layer`. This value is TARGET-GENE-INDEPENDENT (it is a pure function of
    the adata: expression + spatial diffusion), so -- unlike the per-gene betas -- we compute it
    ONCE here for the whole metabolite set.

    We DO NOT re-implement any of it: `received(.)` comes from SpaceTravLR's own
    `init_received_ligands` (which owns the CellChat-collision radii, the export min-expression
    mask, and the narrow eps=1e-3 hard-cutoff kernel) and the summed product comes from
    `SpatialCellularProgramsEstimator.metabolite_interactions`. That is exactly what
    `init_data()` builds into `adata.uns['metabolite_interactions']` at train time, so this
    reproduces what the model saw. Two intentional differences, both harmless for analysis:
      * We compute the diffusion FRESH (never trust an existing `received_ligands*` in `uns`: a
        commot-cached one lacks the metab export genes -- see spaceship.run_commot_). The big
        `cells x ligands` frames it writes to `uns` are removed again unless they were already
        there.
      * We skip the per-target-gene self-exclusion (`parallel_estimators.py:759`) since this
        table is gene-independent; the only effect is that a gene which is ITSELF one of a
        metabolite's transporter genes gets a slightly different value for that one metabolite.

    `metabolites` is the same `{name: [(export, import), ...]}` dict passed to `SpaceShip.fit`;
    `radius`/`contact_distance`/`scale_factor`/`layer` are the run's params (`run_params.json`).
    `adata` must carry the training `layer` and `adata.obsm['spatial']`.

    Returns a DataFrame indexed by `adata.obs_names`, one `metab@<name>` column per metabolite.
    """
    # Lazy import: pulls in torch (via parallel_estimators) -- keep the rest of this module,
    # which is a pure pandas read-out layer, importable without it.
    from SpaceTravLR.models.parallel_estimators import (
        init_received_ligands,
        SpatialCellularProgramsEstimator,
    )

    var = set(adata.var_names)
    # Mirror the estimator's var_names filtering (parallel_estimators.py:749-772), MINUS the
    # per-target-gene self-exclusion (no target gene here). `diffusion_pairs` = every valid pair
    # (deduped across metabolites), which is what must be diffused; `kept` = the summed pairs per
    # surviving metabolite.
    kept = {}
    diffusion_pairs, seen_diff = [], set()
    for name, pairs in metabolites.items():
        keep, seen = [], set()
        for e, i in pairs:
            if e not in var or i not in var:
                continue
            if (e, i) not in seen_diff:
                seen_diff.add((e, i))
                diffusion_pairs.append((e, i))
            if (e, i) in seen:
                continue
            seen.add((e, i))
            keep.append((e, i))
        if keep:
            kept[name] = keep

    if not kept:
        return pd.DataFrame(index=adata.obs_names)

    exports = sorted({e for pairs in kept.values() for e, _ in pairs})
    imports = sorted({i for pairs in kept.values() for _, i in pairs})

    had_tfl = "received_ligands_tfl" in adata.uns
    had_rl = "received_ligands" in adata.uns
    # cell_threshes=None -> received_ligands is set equal to received_ligands_tfl; we read _tfl
    # (the un-thresholded frame the metab path uses).
    init_received_ligands(
        adata, radius=radius, cell_threshes=None,
        contact_distance=contact_distance, scale_factor=scale_factor,
        layer=layer, extra_lr=diffusion_pairs,
    )
    x = SpatialCellularProgramsEstimator.metabolite_interactions(
        adata.uns["received_ligands_tfl"][exports],
        adata.to_df(layer=layer)[imports],
        kept,
    )
    if not had_tfl:
        adata.uns.pop("received_ligands_tfl", None)
    if not had_rl:
        adata.uns.pop("received_ligands", None)

    return x.reindex(adata.obs_names)


def metab_x_to_adata(adata, metabolites, radius, contact_distance=50,
                     scale_factor=100, layer="imputed_count"):
    """Attach the metabolite communication scores (see `compute_metab_x`) to
    `adata.obsm['x_metab']` (cells x metabolites), with the `metab@<name>` labels in
    `adata.uns['x_metab_modulators']` (obsm drops column labels).

    This is ONE shared table (the scores are target-gene-independent), the counterpart to the
    per-gene `beta_<gene>` written by `betas_to_adata`. To get `beta * x` for a gene, multiply
    that gene's metab betas by the matching `x_metab` columns.
    """
    x = compute_metab_x(adata, metabolites, radius, contact_distance, scale_factor, layer)
    adata.obsm["x_metab"] = x.to_numpy()
    adata.uns["x_metab_modulators"] = list(x.columns)
    return adata


def betas_and_metab_x_to_adata(adata, betadata_dir, metabolites, radius=None,
                               contact_distance=50, scale_factor=100, layer="imputed_count",
                               x_adata=None, genes=None, group="metab"):
    """`betas_to_adata` + the metabolite `x` in one pass, so a saved adata carries both the
    per-gene `beta_<gene>` matrices AND the shared `x_metab` communication-score matrix for a
    downstream `beta * x`. This is the pipeline entry point -- it mirrors `betas_to_adata` and
    adds the x half.

    `metabolites` is the `{name: [(export, import), ...]}` dict from the run (`metab_loader`); if
    it is empty/None the x half is skipped and this is exactly `betas_to_adata`. The diffusion is
    run on `x_adata` (defaults to `adata`), which MUST carry the training `layer` +
    `obsm['spatial']` for the exact training cells -- i.e. `input_data/_adata.h5ad`, since the raw
    display adata usually lacks `imputed_count` and may hold a different cell set. Scores are then
    reindexed onto `adata.obs_names` (cells the model didn't fit get NaN, like the betas), so
    `adata.obsm['x_metab']` lines up row-for-row with the `beta_<gene>` matrices.
    `radius`/`contact_distance`/`scale_factor`/`layer` are the run's params (`run_params.json`)
    and must match training. See `compute_metab_x` for the x definition and its one gene-dependent
    edge case.
    """
    betas_to_adata(adata, betadata_dir, genes=genes, group=group)
    if metabolites:
        x = compute_metab_x(
            adata if x_adata is None else x_adata,
            metabolites, radius, contact_distance, scale_factor, layer,
        ).reindex(adata.obs_names)
        adata.obsm["x_metab"] = x.to_numpy()
        adata.uns["x_metab_modulators"] = list(x.columns)
    return adata
