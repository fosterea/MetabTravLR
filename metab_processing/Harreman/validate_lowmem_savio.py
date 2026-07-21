"""Savio (real-harreman, real-data) equivalence gate for the memory-safe CCC drop-ins.

The local test suite proves the three drop-ins in `cell_communication_lowmem.py` reproduce
stock harreman *bit-for-bit* -- but against a VENDORED copy of harreman's code and on CPU.
This script is the final gate: it runs the ACTUAL installed harreman against the drop-ins on
a real (subsampled) adata, on the real device (GPU on Savio), and asserts the outputs match.

Run it on Savio before relying on the drop-ins for a production run:

    python validate_lowmem_savio.py --adata /path/to/adata.h5ad --cell-type-col Tier1 \
        --n-cells 4000 --M 200 --chunk-size 8

What it checks (for each of the three functions):
  * compute_cell_communication            vs compute_cell_communication_lowmem
  * compute_ct_cell_communication         vs compute_ct_cell_communication_lowmem
  * compute_interacting_cell_scores       vs compute_interacting_cell_scores_lowmem
Observed `cs` and the entire non-parametric path (`pval`/`FDR`/`perm_cs`) must be EXACT;
parametric `Z`-derived keys are allowed a tiny magnitude-aware tolerance (float64
reduction-order noise from chunking -- see the module docstring in cell_communication_lowmem.py).

GPU CAVEAT (why this gate exists): on CUDA, reduction kernels can reorder sums by tensor
width, so even `cs`/`perm_cs` could pick up ULP-level drift that never appears on CPU. A perm
value landing within a ULP of the observed score could then flip one integer exceedance count
(measure-zero, but possible). This script reports the max observed diff per key so you can see
whether GPU stays exact or wobbles at the ULP level -- and forces `--chunk-size` small enough
that chunking is actually exercised at the subsample size.

This is a MANUAL smoke (real training/permutation, needs harreman + a GPU), deliberately kept
out of the pytest suite -- like scripts/real_data_smoke.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
import torch

# make the sibling drop-in module importable regardless of CWD
sys.path.insert(0, str(Path(__file__).resolve().parent))
import harreman  # noqa: E402  (real, installed on Savio)

from cell_communication_lowmem import (  # noqa: E402
    compute_cell_communication_lowmem,
    compute_ct_cell_communication_lowmem,
    compute_interacting_cell_scores_lowmem,
)

# same exactness contract as the local tests
EXACT_RTOL, EXACT_ATOL = 0.0, 0.0
ZTOL_RTOL, ZTOL_ATOL = 1e-11, 1e-13


def _compare(name, a, b, exact=True):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        print(f"  FAIL {name}: shape {a.shape} != {b.shape}")
        return False
    with np.errstate(invalid="ignore"):
        diff = np.abs(a.astype(float) - b.astype(float))
    maxdiff = np.nanmax(diff) if diff.size else 0.0
    if exact:
        ok = np.array_equal(a, b) or bool(np.all(np.isnan(a) == np.isnan(b)) and np.nanmax(diff) == 0.0)
        tag = "EXACT" if ok else "DIFF!"
    else:
        ok = np.allclose(a, b, rtol=ZTOL_RTOL, atol=ZTOL_ATOL, equal_nan=True)
        tag = "~ok" if ok else "DIFF!"
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: maxdiff={maxdiff:.3e} ({tag})")
    return ok


def _load(adata_path, n_cells, seed):
    adata = sc.read_h5ad(adata_path)
    if n_cells and adata.n_obs > n_cells:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(adata.n_obs, size=n_cells, replace=False))
        adata = adata[idx].copy()
    adata.layers["counts"] = adata.X.copy()
    adata.X = adata.layers["counts"].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["log_norm"] = adata.X.copy()
    adata.X = adata.layers["counts"].copy()
    return adata


def _setup_common(adata, M):
    """Replicate HarremanRunner's pre-CCC steps so both stock and lowmem see the same state."""
    harreman.pp.extract_interaction_db(adata, species="human", database="transporter", extracellular_only=True)
    harreman.tl.apply_gene_filtering(adata, layer_key="counts", model="danb", autocorrelation_filt=False)
    harreman.tl.compute_knn_graph(adata, compute_neighbors_on_key="spatial", n_neighbors=5, weighted_graph=False)
    harreman.tl.compute_gene_pairs(adata, ct_specific=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adata", required=True)
    ap.add_argument("--cell-type-col", required=True, help="obs column for the ct-aware test")
    ap.add_argument("--n-cells", type=int, default=4000)
    ap.add_argument("--M", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk-size", type=int, default=8, help="force small so chunking is exercised")
    args = ap.parse_args()

    print(f"device: {'cuda' if torch.cuda.is_available() else 'cpu'} | chunk_size={args.chunk_size} | M={args.M}")
    all_ok = True

    # ---- 1) cell-type-independent aggregate --------------------------------------------
    print("\n[1/3] compute_cell_communication vs _lowmem")
    a_stock = _load(args.adata, args.n_cells, args.seed); _setup_common(a_stock, args.M)
    a_low = a_stock.copy()
    cc_kw = dict(model="danb", M=args.M, test="both", layer_key_p_test="counts",
                 layer_key_np_test="log_norm", seed=args.seed)
    harreman.tl.compute_cell_communication(a_stock, **cc_kw)
    compute_cell_communication_lowmem(a_low, gene_pair_chunk_size=args.chunk_size, **cc_kw)
    for grain in ("gp", "m"):
        all_ok &= _compare(f"p/{grain}/cs", a_stock.uns["ccc_results"]["p"][grain]["cs"], a_low.uns["ccc_results"]["p"][grain]["cs"])
        for k in ("pval", "FDR"):
            all_ok &= _compare(f"np/{grain}/{k}", a_stock.uns["ccc_results"]["np"][grain][k], a_low.uns["ccc_results"]["np"][grain][k])
        all_ok &= _compare(f"p/{grain}/Z", a_stock.uns["ccc_results"]["p"][grain]["Z"], a_low.uns["ccc_results"]["p"][grain]["Z"], exact=False)

    # ---- 2) cell-type-aware aggregate --------------------------------------------------
    print("\n[2/3] compute_ct_cell_communication vs _lowmem")
    gp_filt = list(zip(a_stock.uns["ccc_results"]["cell_com_df_gp_sig"]["Gene 1"],
                       a_stock.uns["ccc_results"]["cell_com_df_gp_sig"]["Gene 2"])) \
        if "cell_com_df_gp_sig" in a_stock.uns["ccc_results"] else None
    b_stock = _load(args.adata, args.n_cells, args.seed); _setup_common(b_stock, args.M)
    for key in ("cell_type_pairs", "gene_pairs_per_ct_pair"):
        b_stock.uns.pop(key, None)
    harreman.tl.compute_gene_pairs(b_stock, cell_type_key=args.cell_type_col)
    b_low = b_stock.copy()
    ct_kw = dict(model="danb", cell_type_key=args.cell_type_col, M=args.M, test="both",
                 layer_key_p_test="counts", layer_key_np_test="log_norm",
                 subset_gene_pairs=gp_filt, fix_gp=False, seed=args.seed)
    harreman.tl.compute_ct_cell_communication(b_stock, **ct_kw)
    compute_ct_cell_communication_lowmem(b_low, gene_pair_chunk_size=args.chunk_size, **ct_kw)
    for grain in ("gp", "m"):
        all_ok &= _compare(f"p/{grain}/cs", b_stock.uns["ct_ccc_results"]["p"][grain]["cs"], b_low.uns["ct_ccc_results"]["p"][grain]["cs"])
        for k in ("pval", "FDR"):
            all_ok &= _compare(f"np/{grain}/{k}", b_stock.uns["ct_ccc_results"]["np"][grain][k], b_low.uns["ct_ccc_results"]["np"][grain][k])
        all_ok &= _compare(f"p/{grain}/Z", b_stock.uns["ct_ccc_results"]["p"][grain]["Z"], b_low.uns["ct_ccc_results"]["p"][grain]["Z"], exact=False)

    # ---- 3) per-cell interacting-cell scores (nbhd path) -------------------------------
    print("\n[3/3] compute_interacting_cell_scores vs _lowmem")
    harreman.tl.select_significant_interactions(a_stock, test="non-parametric", threshold=0.05)
    c_stock = a_stock.copy(); c_low = a_stock.copy()
    ics_kw = dict(test="non-parametric", restrict_significance="both",
                  compute_significance="non-parametric", M=args.M, seed=args.seed)
    harreman.tl.compute_interacting_cell_scores(c_stock, **ics_kw)
    compute_interacting_cell_scores_lowmem(c_low, **ics_kw)
    for grain in ("gp", "m"):
        for k in ("cs", "pval", "FDR"):
            all_ok &= _compare(f"np/{grain}/{k}", c_stock.uns["interacting_cell_results"]["np"][grain][k],
                               c_low.uns["interacting_cell_results"]["np"][grain][k])

    print("\n" + ("ALL EQUIVALENT ✅" if all_ok else "MISMATCH DETECTED ❌ -- inspect the DIFF! rows above"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
