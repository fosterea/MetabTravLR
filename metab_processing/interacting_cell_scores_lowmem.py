"""Memory-safe drop-in for ``harreman.tools.compute_interacting_cell_scores``.

WHY: the stock non-parametric (permutation) path allocates, on GPU, the *entire*
permutation null with a per-cell axis:

    perm_cs_gp_a, perm_cs_gp_b : (n_cells, n_gene_pairs, M)
    perm_cs_m_a,  perm_cs_m_b  : (n_cells, n_metabolites, M)

At ~1e5-1e6 cells x ~1e2 pairs x M=1000 this is 10s-100s of GB per array -> OOM.
See DataForClaude/documentation/05_harreman_reference.md sec.5.

WHAT CHANGED (this file vs. harreman's function):
  * The permutation loop no longer stores each permutation. It accumulates the
    exceedance counters incrementally:
        x = sum_over_perms( perm_cs > observed_cs )
    which is exactly what the stock code computed *after* the loop from the giant
    arrays. Peak memory drops from O(cells * pairs * M) to O(cells * pairs).
  * The raw ``perm_cs_a/perm_cs_b`` arrays are NO LONGER written to
    ``adata.uns[...]['np'][...]`` (they were only ever used to derive p-values, and
    storing them is itself huge). Everything downstream (``pval``, ``FDR``, ``cs``,
    ``cs_sig_pval``, ``cs_sig_FDR``) is unchanged and still written.
  * ``check_analytic_null=True`` is rejected (it re-introduces the (cells,pairs,M)
    arrays and also hits a latent NameError in the stock code). Foster calls it with
    False, so this is a no-op for current usage.
  * The parametric ('p') path is reproduced verbatim (it has no M axis / no OOM).

Numerics are otherwise identical: all heavy lifting still uses harreman's own helper
functions, imported from the installed package at runtime (below), so results match
the stock function bit-for-bit for the same seed.

CAVEAT: this was written without a local harreman install (harreman runs on Savio),
so it is UNTESTED here. Validate on Savio, ideally by diffing outputs against the
stock function on a small subset first (see `docstring: sanity check` at bottom).
"""

from __future__ import annotations

import inspect
import time

import numpy as np
import torch
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

import harreman

# --- reuse harreman's own internals so the math is identical --------------------------
_MOD = inspect.getmodule(harreman.tools.compute_interacting_cell_scores)


def _need(name):
    fn = getattr(_MOD, name, None)
    if fn is None:
        raise ImportError(
            f"Could not find helper '{name}' in {_MOD.__name__}. harreman's internal "
            f"layout may have changed; update interacting_cell_scores_lowmem.py."
        )
    return fn


counts_from_anndata = _need("counts_from_anndata")
standardize_counts = _need("standardize_counts")
make_weights_non_redundant = _need("make_weights_non_redundant")
compute_metabolite_cs = _need("compute_metabolite_cs")
compute_p_int_cell_results_no_ct = _need("compute_p_int_cell_results_no_ct")


def compute_interacting_cell_scores_lowmem(
    adata,
    center_counts_for_np_test: bool = False,
    test: str = "both",
    restrict_significance: str = "both",
    compute_significance: str = "both",
    M: int = 1000,
    seed: int = 42,
    check_analytic_null: bool = False,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    verbose: bool = False,
):
    """Memory-safe equivalent of ``harreman.tools.compute_interacting_cell_scores``.

    Same signature and same ``adata.uns['interacting_cell_results']`` outputs, minus the
    raw permutation arrays. See module docstring for the (small) list of differences.
    """
    start = time.time()
    if verbose:
        print("[lowmem] Computing gene pair and metabolite scores...")

    if check_analytic_null:
        raise NotImplementedError(
            "check_analytic_null=True is not supported in the low-memory version "
            "(it re-introduces the (cells, pairs, M) tensors this patch removes)."
        )

    adata.uns["interacting_cell_results"] = {}

    model = adata.uns["model"]
    mean = adata.uns["mean"]

    if test not in ["both", "parametric", "non-parametric"]:
        raise ValueError('The "test" variable should be one of ["both", "parametric", "non-parametric"].')
    if restrict_significance is not None and restrict_significance not in ["both", "gene pairs", "metabolites"]:
        raise ValueError('The "restrict_significance" variable should be one of ["both", "gene pairs", "metabolites"].')
    if compute_significance is not None and compute_significance not in ["both", "parametric", "non-parametric"]:
        raise ValueError('The "compute_significance" variable should be one of ["both", "parametric", "non-parametric"].')

    import pandas as pd  # local, matches harreman's usage

    sample_specific = "sample_key" in adata.uns

    layer_key_p_test = adata.uns.get("layer_key_p_test", None)
    layer_key_np_test = adata.uns.get("layer_key_np_test", None)
    use_raw = (layer_key_p_test == "use_raw") and (layer_key_np_test == "use_raw")

    gene_pairs = adata.uns.get("gene_pairs", None)
    gene_pairs_per_metabolite = adata.uns["gene_pairs_per_metabolite"]

    def to_tuple(x):
        if isinstance(x, list):
            return tuple(to_tuple(i) for i in x)
        return x

    metabolite_gene_pair_df = pd.DataFrame.from_dict(gene_pairs_per_metabolite, orient="index").reset_index()
    metabolite_gene_pair_df = metabolite_gene_pair_df.rename(columns={"index": "metabolite"})
    metabolite_gene_pair_df["gene_pair"] = metabolite_gene_pair_df["gene_pair"].apply(
        lambda arr: [(to_tuple(gp[0]), to_tuple(gp[1])) for gp in arr]
    )
    metabolite_gene_pair_df["gene_type"] = metabolite_gene_pair_df["gene_type"].apply(
        lambda arr: [(to_tuple(gt[0]), to_tuple(gt[1])) for gt in arr]
    )
    metabolite_gene_pair_df = pd.concat(
        [
            metabolite_gene_pair_df["metabolite"],
            metabolite_gene_pair_df.explode("gene_pair")["gene_pair"],
            metabolite_gene_pair_df.explode("gene_type")["gene_type"],
        ],
        axis=1,
    ).reset_index(drop=True)

    if "LR_database" in adata.uns:
        LR_database = adata.uns["LR_database"]
        df_merged = pd.merge(metabolite_gene_pair_df, LR_database, left_on="metabolite", right_on="interaction_name", how="left")
        LR_df = df_merged.dropna(subset=["pathway_name"])
        metabolite_gene_pair_df["metabolite"][metabolite_gene_pair_df.metabolite.isin(LR_df.metabolite)] = LR_df["pathway_name"]

    if restrict_significance in ["both", "gene pairs"]:
        cell_com_gp_df = adata.uns["ccc_results"]["cell_com_df_gp_sig"].copy()
        cell_com_gp_df[["Gene 1", "Gene 2"]] = cell_com_gp_df[["Gene 1", "Gene 2"]].applymap(
            lambda x: tuple(x) if isinstance(x, list) else x
        )
        gene_pairs_set = set([tuple(x) for x in cell_com_gp_df[["Gene 1", "Gene 2"]].values])
        metabolite_gene_pair_df = metabolite_gene_pair_df[metabolite_gene_pair_df["gene_pair"].isin(gene_pairs_set)]

    if restrict_significance in ["both", "metabolites"]:
        cell_com_m_df = adata.uns["ccc_results"]["cell_com_df_m_sig"].copy()
        metabolite_set = set(cell_com_m_df["Metabolite"].values)
        metabolite_gene_pair_df = metabolite_gene_pair_df[metabolite_gene_pair_df["metabolite"].isin(metabolite_set)]

    genes = adata.uns["genes"]
    gene_pairs_sig = []
    if gene_pairs:
        for g1, g2 in gene_pairs:
            g1 = tuple(g1) if isinstance(g1, list) else g1
            g2 = tuple(g2) if isinstance(g2, list) else g2
            if not metabolite_gene_pair_df[metabolite_gene_pair_df["gene_pair"] == (g1, g2)].empty:
                gene_pairs_sig.append((g1, g2))
    adata.uns["gene_pairs_sig"] = gene_pairs_sig

    gene_pairs_sig_ind = []
    for g1, g2 in gene_pairs_sig:
        idx1 = tuple([genes.index(g) for g in g1]) if isinstance(g1, tuple) else genes.index(g1)
        idx2 = tuple([genes.index(g) for g in g2]) if isinstance(g2, tuple) else genes.index(g2)
        gene_pairs_sig_ind.append((idx1, idx2))
    adata.uns["gene_pairs_sig_ind"] = gene_pairs_sig_ind

    if "barcode_key" in adata.uns:
        barcode_key = adata.uns["barcode_key"]
        cells = pd.Series(adata.obs[barcode_key].tolist())
    else:
        cells = adata.obs_names if not use_raw else adata.raw.obs_names

    weights = make_weights_non_redundant(adata.obsp["weights"]).tocoo()
    weights = torch.sparse_coo_tensor(
        torch.tensor(np.vstack((weights.row, weights.col)), dtype=torch.long, device=device),
        torch.tensor(weights.data, dtype=torch.float64, device=device),
        torch.Size(weights.shape),
        device=device,
    )

    gene_pair_dict = {}
    for metabolite, group in metabolite_gene_pair_df.groupby("metabolite"):
        idxs = group["gene_pair"].apply(lambda gp: gene_pairs_sig.index(gp) if gp in gene_pairs_sig else None).dropna().tolist()
        idxs = [int(ind) for ind in idxs if ind is not None]
        if idxs:
            gene_pair_dict[metabolite] = idxs
    metabolites = list(gene_pair_dict.keys())
    adata.uns["metabolites"] = metabolites

    gene_pairs_sig_names = [
        "_".join("_".join(g) if isinstance(g, tuple) else g for g in gp) for gp in gene_pairs_sig
    ]
    adata.uns["gene_pairs_sig_names"] = gene_pairs_sig_names

    # ================================ parametric ('p') ================================
    # (verbatim from harreman; no per-permutation / M axis, so no OOM here)
    if test in ["parametric", "both"]:
        if verbose:
            print("[lowmem] Running the parametric test...")
        adata.uns["interacting_cell_results"]["p"] = {"gp": {}, "m": {}}

        Wtot2 = torch.tensor((weights.values() ** 2).sum(), device=device)

        counts = counts_from_anndata(adata[cells, genes], layer_key_p_test, dense=True)
        counts = torch.tensor(counts, dtype=torch.float64, device=device)
        num_umi = counts.sum(dim=0)

        counts_1, counts_2 = _prep_counts_1_2(counts, gene_pairs_sig_ind, mean)
        counts_1 = standardize_counts(adata, counts_1, model, num_umi, sample_specific)
        counts_2 = standardize_counts(adata, counts_2, model, num_umi, sample_specific)

        WX2t = torch.sparse.mm(weights, counts_2.T)
        WtX2t = torch.sparse.mm(weights.transpose(0, 1), counts_2.T)
        cs_gp = (counts_1.T * WX2t) + (counts_1.T * WtX2t)
        same_gene_mask = torch.tensor([g1 == g2 for g1, g2 in gene_pairs_sig], device=device)
        cs_gp[:, same_gene_mask] = cs_gp[:, same_gene_mask] / 2
        adata.uns["interacting_cell_results"]["p"]["gp"]["cs"] = cs_gp.detach().cpu().numpy()

        cs_m = compute_metabolite_cs(cs_gp, gene_pair_dict, interacting_cell_scores=True)
        adata.uns["interacting_cell_results"]["p"]["m"]["cs"] = cs_m.detach().cpu().numpy()

        if compute_significance in ["parametric", "both"]:
            WX1t = torch.sparse.mm(weights, counts_1.T)
            WtX1t = torch.sparse.mm(weights.transpose(0, 1), counts_1.T)
            eg2_a = (WX1t + WtX1t).pow(2)
            eg2_b = (WX2t + WtX2t).pow(2)
            eg2s_gp = (eg2_a, eg2_b)

            Z_gp, Z_m = compute_p_int_cell_results_no_ct(cs_gp, cs_m, gene_pairs_sig_ind, Wtot2, eg2s_gp, gene_pair_dict)
            Z_gp_np = Z_gp.detach().cpu().numpy()
            Z_m_np = Z_m.detach().cpu().numpy()
            Z_pvals_gp = norm.sf(Z_gp_np)
            Z_pvals_m = norm.sf(Z_m_np)
            FDR_gp = multipletests(Z_pvals_gp.flatten(), method="fdr_bh")[1].reshape(Z_pvals_gp.shape)
            FDR_m = multipletests(Z_pvals_m.flatten(), method="fdr_bh")[1].reshape(Z_pvals_m.shape)

            p_gp = adata.uns["interacting_cell_results"]["p"]["gp"]
            p_m = adata.uns["interacting_cell_results"]["p"]["m"]
            p_gp["Z"], p_gp["Z_pval"], p_gp["Z_FDR"] = Z_gp_np, Z_pvals_gp, FDR_gp
            p_m["Z"], p_m["Z_pval"], p_m["Z_FDR"] = Z_m_np, Z_pvals_m, FDR_m

            _write_sig_masks(p_gp, p_m, Z_pvals_gp, Z_pvals_m, FDR_gp, FDR_m)

        if verbose:
            print("[lowmem] Parametric test finished.")

    # ============================ non-parametric ('np') ==============================
    # THE MEMORY FIX LIVES HERE.
    if test in ["non-parametric", "both"]:
        if verbose:
            print("[lowmem] Running the non-parametric test...")
        adata.uns["interacting_cell_results"]["np"] = {"gp": {}, "m": {}}

        counts = counts_from_anndata(adata[cells, genes], layer_key_np_test, dense=True)
        counts = torch.tensor(counts, dtype=torch.float64, device=device)

        counts_1, counts_2 = _prep_counts_1_2(counts, gene_pairs_sig_ind, mean)

        if center_counts_for_np_test:
            num_umi = counts.sum(dim=0)
            counts_1 = standardize_counts(adata, counts_1, model, num_umi, sample_specific)
            counts_2 = standardize_counts(adata, counts_2, model, num_umi, sample_specific)

        n_cells = counts_1.shape[1]
        n_gp = counts_1.shape[0]
        same_gene_mask = torch.tensor([g1 == g2 for g1, g2 in gene_pairs_sig], device=device)

        if center_counts_for_np_test and test == "both":
            adata.uns["interacting_cell_results"]["np"]["gp"]["cs"] = np.array(adata.uns["interacting_cell_results"]["p"]["gp"]["cs"])
            adata.uns["interacting_cell_results"]["np"]["m"]["cs"] = np.array(adata.uns["interacting_cell_results"]["p"]["m"]["cs"])
        else:
            WX2t = torch.sparse.mm(weights, counts_2.T)
            WtX2t = torch.sparse.mm(weights.transpose(0, 1), counts_2.T)
            cs_gp = (counts_1.T * WX2t) + (counts_1.T * WtX2t)
            cs_gp[:, same_gene_mask] = cs_gp[:, same_gene_mask] / 2
            adata.uns["interacting_cell_results"]["np"]["gp"]["cs"] = cs_gp.detach().cpu().numpy()
            cs_m = compute_metabolite_cs(cs_gp, gene_pair_dict, interacting_cell_scores=True)
            adata.uns["interacting_cell_results"]["np"]["m"]["cs"] = cs_m.detach().cpu().numpy()

        if compute_significance in ["non-parametric", "both"]:
            # observed scores as tensors (robust to either branch above)
            cs_gp = torch.as_tensor(
                np.asarray(adata.uns["interacting_cell_results"]["np"]["gp"]["cs"]), dtype=torch.float64, device=device
            )
            cs_m = torch.as_tensor(
                np.asarray(adata.uns["interacting_cell_results"]["np"]["m"]["cs"]), dtype=torch.float64, device=device
            )
            n_m = cs_m.shape[1]

            # --- CHANGED: incremental exceedance counters instead of (cells, *, M) arrays
            x_gp_a = torch.zeros((n_cells, n_gp), dtype=torch.float64, device=device)
            x_gp_b = torch.zeros_like(x_gp_a)
            x_m_a = torch.zeros((n_cells, n_m), dtype=torch.float64, device=device)
            x_m_b = torch.zeros_like(x_m_a)

            torch.manual_seed(seed)
            for _ in tqdm(range(M), desc="[lowmem] Permutation test", disable=not verbose):
                idx = torch.randperm(n_cells, device=device)

                # permute the "receiver" (counts_2), keep "sender" (counts_1) — arm a
                c1_perm_a = counts_1.clone()
                c2_perm_a = counts_2[:, idx]
                c1_perm_a[same_gene_mask] = counts_1[same_gene_mask, :][:, idx]
                WX2t_a = torch.sparse.mm(weights, c2_perm_a.T)
                WtX2t_a = torch.sparse.mm(weights.transpose(0, 1), c2_perm_a.T)
                cs_a = (c1_perm_a.T * WX2t_a) + (c1_perm_a.T * WtX2t_a)
                cs_a[:, same_gene_mask] = cs_a[:, same_gene_mask] / 2
                cs_m_a = compute_metabolite_cs(cs_a, gene_pair_dict, interacting_cell_scores=True)
                x_gp_a += (cs_a > cs_gp).to(torch.float64)
                x_m_a += (cs_m_a > cs_m).to(torch.float64)

                # permute the "sender" (counts_1), keep "receiver" (counts_2) — arm b
                c2_perm_b = counts_2.clone()
                c1_perm_b = counts_1[:, idx]
                c2_perm_b[same_gene_mask] = counts_2[same_gene_mask, :][:, idx]
                WX2t_b = torch.sparse.mm(weights, c2_perm_b.T)
                WtX2t_b = torch.sparse.mm(weights.transpose(0, 1), c2_perm_b.T)
                cs_b = (c1_perm_b.T * WX2t_b) + (c1_perm_b.T * WtX2t_b)
                cs_b[:, same_gene_mask] = cs_b[:, same_gene_mask] / 2
                cs_m_b = compute_metabolite_cs(cs_b, gene_pair_dict, interacting_cell_scores=True)
                x_gp_b += (cs_b > cs_gp).to(torch.float64)
                x_m_b += (cs_m_b > cs_m).to(torch.float64)

            pvals_gp_a = (x_gp_a + 1) / (M + 1)
            pvals_gp_b = (x_gp_b + 1) / (M + 1)
            pvals_m_a = (x_m_a + 1) / (M + 1)
            pvals_m_b = (x_m_b + 1) / (M + 1)

            pvals_gp = torch.where(pvals_gp_a > pvals_gp_b, pvals_gp_a, pvals_gp_b).cpu().numpy()
            pvals_m = torch.where(pvals_m_a > pvals_m_b, pvals_m_a, pvals_m_b).cpu().numpy()

            np_gp = adata.uns["interacting_cell_results"]["np"]["gp"]
            np_m = adata.uns["interacting_cell_results"]["np"]["m"]
            np_gp["pval"] = pvals_gp
            np_gp["FDR"] = multipletests(pvals_gp.flatten(), method="fdr_bh")[1].reshape(pvals_gp.shape)
            np_m["pval"] = pvals_m
            np_m["FDR"] = multipletests(pvals_m.flatten(), method="fdr_bh")[1].reshape(pvals_m.shape)

            _write_sig_masks(np_gp, np_m, pvals_gp, pvals_m, np_gp["FDR"], np_m["FDR"])

        if verbose:
            print("[lowmem] Non-parametric test finished.")

    if verbose:
        print("[lowmem] Finished in %.3f seconds" % (time.time() - start))
    return


def _prep_counts_1_2(counts, gene_pairs_sig_ind, mean):
    """Build the (n_gene_pairs, n_cells) counts_1 / counts_2 stacks. Verbatim logic from
    harreman (handles heterodimer tuple/list indices)."""
    counts_1, counts_2 = [], []
    for (idx1, idx2) in gene_pairs_sig_ind:
        if isinstance(idx1, (tuple, list)):
            c1 = counts[list(idx1), :].mean(dim=0) if mean == "algebraic" else torch.exp(torch.log(counts[list(idx1), :] + 1e-8).mean(dim=0))
        else:
            c1 = counts[idx1, :]
        if isinstance(idx2, (tuple, list)):
            c2 = counts[list(idx2), :].mean(dim=0) if mean == "algebraic" else torch.exp(torch.log(counts[list(idx2), :] + 1e-8).mean(dim=0))
        else:
            c2 = counts[idx2, :]
        counts_1.append(c1)
        counts_2.append(c2)
    return torch.stack(counts_1), torch.stack(counts_2)


def _write_sig_masks(gp_dict, m_dict, pval_or_none_gp, pval_or_none_m, fdr_gp, fdr_m):
    """Write the cs_sig_pval / cs_sig_FDR masked arrays (verbatim behavior)."""
    # p-value masks (only when a p-value array is available)
    if pval_or_none_gp is not None:
        for key, arr, mask_src in (
            ("cs_sig_pval", "cs", pval_or_none_gp),
        ):
            m = mask_src < 0.05
            cs = gp_dict["cs"].copy()
            cs[~m] = np.nan
            gp_dict[key] = cs
        m = pval_or_none_m < 0.05
        cs = m_dict["cs"].copy()
        cs[~m] = np.nan
        m_dict["cs_sig_pval"] = cs
    # FDR masks
    m = fdr_gp < 0.05
    cs = gp_dict["cs"].copy()
    cs[~m] = np.nan
    gp_dict["cs_sig_FDR"] = cs
    m = fdr_m < 0.05
    cs = m_dict["cs"].copy()
    cs[~m] = np.nan
    m_dict["cs_sig_FDR"] = cs


# --- sanity check (run on Savio, small subset) ----------------------------------------
# To trust this before a full run, compare against the stock function on a tiny adata:
#
#   import harreman, numpy as np
#   from interacting_cell_scores_lowmem import compute_interacting_cell_scores_lowmem
#   sub = adata[:2000].copy()               # small enough for the stock version
#   a = sub.copy(); b = sub.copy()
#   harreman.tools.compute_interacting_cell_scores(a, test='both', M=200, seed=0)
#   compute_interacting_cell_scores_lowmem(b, test='both', M=200, seed=0)
#   for t in ['p','np']:
#       for g in ['gp','m']:
#           np.testing.assert_allclose(a.uns['interacting_cell_results'][t][g]['cs'],
#                                      b.uns['interacting_cell_results'][t][g]['cs'])
#   # p-values use RNG; with the same seed/device they should match closely.
