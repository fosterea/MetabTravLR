"""Beta-analysis helper: read trained metabolite betas back out of `betadata/` and
aggregate them into interpretable tables (metabolite x gene[, cell_type] scores, then
signed gene-set scores). This is the read-out side of MetabTravLR -- we analyze the
model's learned coefficients directly, no perturbation.

Pipeline: `read_metab_beta_summary` (per-gene betadata parquet -> per-(gene, gene-pair)
beta stats) -> `aggregate_to_metabolite` (roll gene-pairs up to metabolites via a
`metab_loader.load_metabolite_selection` mapping, optionally C_np-weighted) ->
`gene_set_score` (signed score of each metabolite against a labeled target gene set).
`gene_pair_cnp_weights` is a thin convenience that builds the optional weights for the
aggregation step from harreman's communication scores.

Pandas-only; no torch/scanpy needed to read betadata back out.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def read_metab_beta_summary(betadata_dir, genes=None, obs=None, cell_type_col=None) -> pd.DataFrame:
    """Summarize the metabolite (`beta_<export>@<import>`) columns of `{gene}_betadata.
    parquet` files, ACROSS CELLS, streaming gene-by-gene so memory stays bounded (data
    can be 100k+ cells x many genes -- we never materialize a per-cell long frame).

    NOTE: reads parquet with plain `pd.read_parquet` + a column filter, NOT
    `beta.py::BetaFrame` -- that class classifies modulator columns purely by separator
    (`$` = L-R, `#` = L-TF, none = TF) and would misclassify `@` metabolite columns.

    Parameters
    ----------
    betadata_dir : directory containing `{gene}_betadata.parquet` files.
    genes : restrict to these genes; a requested gene with no betadata file is skipped
        (printed, not an error). Default (None) = every `*_betadata.parquet` present.
    obs, cell_type_col : if BOTH given, group stats by `obs[cell_type_col]` (aligned to
        each gene's betadata index by cell name) instead of pooling all cells -- this is
        the "aggregate across cell types for a tier" case, where the tier is whichever
        `adata.obs` column is passed. If either is omitted, cells are pooled (no
        `cell_type` column in the output).

    Returns
    -------
    DataFrame, one row per (gene, export, import[, cell_type]):
    `gene, export, import, pair, [cell_type,] mean, std, n, frac_nonzero`.
    `pair` = `"<export>@<import>"`; `frac_nonzero` = fraction of cells with beta != 0.
    """
    betadata_dir = Path(betadata_dir)
    group_by_ct = obs is not None and cell_type_col is not None

    if genes is None:
        paths = sorted(betadata_dir.glob("*_betadata.parquet"))
    else:
        paths = []
        for g in genes:
            p = betadata_dir / f"{g}_betadata.parquet"
            if p.is_file():
                paths.append(p)
            else:
                print(f"read_metab_beta_summary: no betadata for gene {g!r}, skipping")

    rows = []
    for path in paths:
        gene = path.name[: -len("_betadata.parquet")]
        schema_cols = pq.read_schema(path).names
        metab_cols = [c for c in schema_cols if "@" in c]
        if not metab_cols:
            continue
        df = pd.read_parquet(path, columns=metab_cols)

        groups = df.groupby(obs.loc[df.index, cell_type_col]) if group_by_ct else [(None, df)]

        for ct_val, gdf in groups:
            for col in metab_cols:
                vals = gdf[col]
                pair = col[len("beta_"):] if col.startswith("beta_") else col
                export, import_ = pair.split("@", 1)
                row = {
                    "gene": gene,
                    "export": export,
                    "import": import_,
                    "pair": pair,
                }
                if group_by_ct:
                    row["cell_type"] = ct_val
                row.update({
                    "mean": vals.mean(),
                    "std": vals.std(),
                    "n": len(vals),
                    "frac_nonzero": float((vals != 0).mean()),
                })
                rows.append(row)

    cols = ["gene", "export", "import", "pair"]
    if group_by_ct:
        cols.append("cell_type")
    cols += ["mean", "std", "n", "frac_nonzero"]
    return pd.DataFrame(rows, columns=cols)


def aggregate_to_metabolite(pair_summary, selection, weights=None) -> pd.DataFrame:
    """Roll a per-gene-pair beta summary (from `read_metab_beta_summary`) up to the
    metabolite level, using `selection` (`{metabolite: [(g1, g2), ...]}`, as returned by
    `metab_processing.metab_loader.load_metabolite_selection`).

    For each metabolite, gathers all its gene pairs in EITHER orientation (unordered
    `{a, b}` matches `pair_summary` rows with `(export, import)` in `{(a, b), (b, a)}`)
    that are present in `pair_summary`, and combines their `mean` beta into a per-
    (metabolite, gene[, cell_type]) `score`:
      - default: simple mean across the metabolite's matching pair rows.
      - `weights` given (`dict[frozenset({g1, g2}) -> float]`, e.g. from
        `gene_pair_cnp_weights`): weighted mean, weighting each pair row by its
        unordered-pair weight. A pair absent from `weights` is SKIPPED entirely (not
        just zero-weighted) for that metabolite -- excluded from both the average and
        `n_pairs`. A pair absent from `pair_summary` altogether is simply never matched.

    Returns
    -------
    DataFrame with columns `metabolite, gene[, cell_type], score, n_pairs` (`n_pairs` =
    how many `pair_summary` rows contributed to that row's score).
    """
    has_ct = "cell_type" in pair_summary.columns
    group_cols = ["gene", "cell_type"] if has_ct else ["gene"]
    out_cols = ["metabolite"] + group_cols + ["score", "n_pairs"]

    if pair_summary.empty:
        return pd.DataFrame(columns=out_cols)

    ps = pair_summary.copy()
    ps["_pair_key"] = [frozenset((e, i)) for e, i in zip(ps["export"], ps["import"])]

    rows = []
    for metab, pairs in selection.items():
        keys = list(dict.fromkeys(frozenset((g1, g2)) for g1, g2 in pairs))
        sub = ps[ps["_pair_key"].isin(keys)]
        if weights is not None:
            sub = sub[sub["_pair_key"].isin(weights.keys())]
        if sub.empty:
            continue
        for group_vals, gdf in sub.groupby(group_cols, dropna=False):
            if not isinstance(group_vals, tuple):
                group_vals = (group_vals,)
            if weights is not None:
                w = gdf["_pair_key"].map(weights).astype(float)
                wsum = w.sum()
                score = float((gdf["mean"] * w).sum() / wsum) if wsum != 0 else np.nan
            else:
                score = float(gdf["mean"].mean())
            row = dict(zip(group_cols, group_vals))
            row["metabolite"] = metab
            row["score"] = score
            row["n_pairs"] = int(len(gdf))
            rows.append(row)

    return pd.DataFrame(rows, columns=out_cols)


def gene_set_score(metab_summary, gene_sets) -> pd.DataFrame:
    """Signed gene-set score per metabolite[, cell_type], from `aggregate_to_metabolite`'s
    output. `gene_sets = {label: [genes]}`, e.g. `{'positive': [...], 'negative': [...]}`.

    Returns one column per label = nan-aware mean `score` over that label's genes
    (genes in a label that are absent from `metab_summary` are simply not scored, not an
    error; a metabolite scored for none of a label's genes gets NaN for that label),
    PLUS a `signed` column = `positive - negative` when the label set is exactly
    `{'positive', 'negative'}`.

    Sorted descending by `signed` if present, else by the sole label's score if there is
    exactly one label; otherwise left in natural (groupby) order.
    """
    has_ct = "cell_type" in metab_summary.columns
    index_cols = ["metabolite", "cell_type"] if has_ct else ["metabolite"]
    labels = list(gene_sets.keys())

    if metab_summary.empty:
        return pd.DataFrame(columns=index_cols + labels)

    pivot = metab_summary.pivot_table(index=index_cols, columns="gene", values="score", aggfunc="mean")

    out = pivot.index.to_frame(index=False)
    for label in labels:
        present = [g for g in gene_sets[label] if g in pivot.columns]
        out[label] = pivot[present].mean(axis=1, skipna=True).to_numpy() if present else np.nan

    if set(labels) == {"positive", "negative"}:
        out["signed"] = out["positive"] - out["negative"]
        out = out.sort_values("signed", ascending=False)
    elif len(labels) == 1:
        out = out.sort_values(labels[0], ascending=False)

    return out.reset_index(drop=True)


def gene_pair_cnp_weights(easy_download_path, tier=None, agg="max") -> dict:
    """Build `{frozenset({Gene1, Gene2}): C_np}` communication-score weights for
    `aggregate_to_metabolite(weights=...)`, from harreman output.

    Implementation note: rather than calling `summarize_harreman_folder` directly (which
    needs the FULL `harreman_outputs` folder -- `harreman_network.json` plus every tier's
    CSVs, to build both the master and gene-pair tables), this reads the already-written
    flat `gene_pair_summary.csv` (written by `harreman_summary.main()` /
    `genepairs.to_csv(...)`, columns include `tier, gene1, gene2, C_np`) when one is
    found -- simplest, robust, single-file. `easy_download_path` may point directly at
    that CSV, at a folder containing `summary/gene_pair_summary.csv` or
    `gene_pair_summary.csv`, or (fallback, if none of those exist) at a full
    `harreman_outputs` folder, in which case `summarize_harreman_folder` is called.

    Parameters
    ----------
    tier : restrict to this `tier` value (e.g. `"Tier1"`, or `"global"` for the
        cell-type-independent table); `None` = use all rows (all tiers pooled).
    agg : `'max'` or `'mean'` -- how to reduce duplicate (Gene1, Gene2) rows (e.g. the
        same pair significant in multiple cell-type interfaces) to one weight.
    """
    if agg not in ("max", "mean"):
        raise ValueError("agg must be 'max' or 'mean'")

    path = Path(easy_download_path)
    if path.is_file():
        genepairs = pd.read_csv(path)
    elif (path / "summary" / "gene_pair_summary.csv").is_file():
        genepairs = pd.read_csv(path / "summary" / "gene_pair_summary.csv")
    elif (path / "gene_pair_summary.csv").is_file():
        genepairs = pd.read_csv(path / "gene_pair_summary.csv")
    else:
        from metab_processing.Harreman.harreman_summary import summarize_harreman_folder
        _, genepairs = summarize_harreman_folder(path)

    if tier is not None:
        genepairs = genepairs[genepairs["tier"] == tier]
    if genepairs.empty:
        return {}

    keys = [frozenset((g1, g2)) for g1, g2 in zip(genepairs["gene1"], genepairs["gene2"])]
    grouped = pd.Series(genepairs["C_np"].to_numpy(), index=keys).groupby(level=0)
    reduced = grouped.max() if agg == "max" else grouped.mean()
    return reduced.to_dict()
