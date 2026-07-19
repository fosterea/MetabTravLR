"""Read trained betas back out of `betadata/` and summarize them per tier.

A tier is an `adata.obs` column of cell-type labels (e.g. `Tier1`). For each tier we
group cells by their label and take the mean/std of each modulator's beta.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SEPARATORS = {"@": "metab", "$": "lr", "#": "ltf"}


def _group(modulator):
    """Modulator group: 'metab', 'lr', 'ltf', or 'tf' (no separator)."""
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


def write_gene_pairs(betadata_dir, obs, tiers, outdir, genes=None):
    """Write `<outdir>/<tier>/gene_pairs.csv`: metabolite-pair betas per (gene, pair, cell type).

    Both orientations of a heterotypic pair stay as separate rows. Returns {tier: DataFrame}.
    """
    out = {}
    for tier in tiers:
        stats = tier_means(betadata_dir, obs, tier, genes, group="metab")
        stats[["export", "import"]] = stats["modulator"].str.split("@", n=1, expand=True)
        stats = stats.rename(columns={"modulator": "pair"})
        stats = stats[["gene", "export", "import", "pair", "cell_type", "mean", "std", "n"]]

        tier_dir = Path(outdir) / tier
        tier_dir.mkdir(parents=True, exist_ok=True)
        stats.to_csv(tier_dir / "gene_pairs.csv", index=False)
        out[tier] = stats
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
