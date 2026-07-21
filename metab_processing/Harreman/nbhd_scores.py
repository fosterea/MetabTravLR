"""Per-cell neighborhood scores (harreman interacting-cell scores) and per-tier summaries.

The score is harreman's: for cell i and gene pair (g1, g2), its own expression of g1 times its
neighbors' expression of g2, symmetrized. Metabolite scores sum a metabolite's gene pairs.
The scores are cell-type-independent, so every tier is a groupby over the same matrices.
"""

import numpy as np
import pandas as pd

from cell_communication_lowmem import compute_interacting_cell_scores_lowmem

GRAIN_COL = {'m': 'metabolite', 'gp': 'gene_pair'}


def compute_nbhd_scores(adata, M=1000, seed=42, verbose=False,
                         gene_pair_chunk_size=None, metabolite_chunk_size=None):
    """Store per-cell scores in adata.uns['interacting_cell_results']['np'].

    ``gene_pair_chunk_size``/``metabolite_chunk_size`` are forwarded to
    `compute_interacting_cell_scores_lowmem` (CU-E, Option B chunking); both default to
    ``None`` (adaptive sizing), matching prior behavior byte-for-byte.
    """
    compute_interacting_cell_scores_lowmem(
        adata,
        test='non-parametric',
        restrict_significance='both',
        compute_significance='non-parametric',
        M=M,
        seed=seed,
        verbose=verbose,
        gene_pair_chunk_size=gene_pair_chunk_size,
        metabolite_chunk_size=metabolite_chunk_size,
    )


def summarize_nbhd_scores(adata, cell_type_col, grain='m', alpha=0.05):
    """One row per (cell type, metabolite or gene pair), summarizing the per-cell scores."""
    res = adata.uns['interacting_cell_results']['np'][grain]
    cs, pval, fdr = res['cs'], res['pval'], res['FDR']
    labels = adata.uns['metabolites' if grain == 'm' else 'gene_pairs_sig_names']

    sig = (fdr < alpha) & (cs > 0)  # harreman's 'selected' convention
    overall_frac = sig.mean(axis=0)
    neg_log10_pval = -np.log10(pval)

    groups = adata.obs[cell_type_col].astype(str).values
    rows = []
    for label in np.unique(groups):
        mask = groups == label
        n_sig = sig[mask].sum(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_cs_sig = np.where(
                n_sig > 0,
                np.where(sig[mask], cs[mask], 0.0).sum(axis=0) / np.maximum(n_sig, 1),
                np.nan,
            )
            frac_sig = n_sig / mask.sum()
            log2_enrichment = np.log2(frac_sig / overall_frac)
        rows.append(pd.DataFrame({
            'cell_type': label,
            GRAIN_COL[grain]: labels,
            'n_cells': mask.sum(),
            'frac_sig': frac_sig,
            'mean_cs': cs[mask].mean(axis=0),
            'mean_cs_sig': mean_cs_sig,
            'mean_neg_log10_pval': neg_log10_pval[mask].mean(axis=0),
            'log2_enrichment': log2_enrichment,
        }))

    return pd.concat(rows, ignore_index=True)
