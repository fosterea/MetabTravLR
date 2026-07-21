"""Savio (real-harreman, real-data, GPU) equivalence gate for the memory-safe CCC drop-ins.

The local test suite proves the three drop-ins in `cell_communication_lowmem.py` reproduce
stock harreman *bit-for-bit* -- but against a VENDORED copy of harreman's code and on CPU.
This script is the final gate: it runs the ACTUAL installed harreman against the drop-ins on
a real (subsampled) adata, on the real device (GPU on Savio), and compares outputs.

    python validate_lowmem_savio.py --adata /path/to/adata.h5ad --cell-type-col Tier1 \
        --n-cells 8000 --M 200 --chunk-size 8

The per-cell nbhd path (section [3/3]) chunks TWO axes (gene-pair + metabolite, CU-E); both
default to --chunk-size but can be forced independently via --nbhd-gp-chunk-size /
--nbhd-m-chunk-size. Its non-parametric cs/pval/FDR (both grains) must be EXACT vs stock; to
truly exercise the ≥600k-cell OOM fix, run with a large --n-cells (and small nbhd chunks) so
STOCK OOMs while the chunked drop-in survives (reported as "fix demonstrated").

EXACTNESS CONTRACT (this is the important part -- read before reacting to a "DIFF"):
  * NON-PARAMETRIC path (`pval`/`FDR`, and per-cell `cs`): must be EXACTLY bit-identical
    (maxdiff 0.0). This is what `select_significant_interactions(test='non-parametric')`
    gates your production results on. A nonzero diff here is a REAL problem.
  * PARAMETRIC path (`cs`/`Z`/`Z_*`): compared with a magnitude-aware tolerance, NOT exact.
    On CUDA, `.sum(dim=0)` reduction kernels reorder accumulation by tensor width, so the
    chunked parametric score drifts from stock's full-width sum by a few ULPs of float64
    (~1e-16 relative). This is expected and scientifically irrelevant -- it is off the
    gating path. (On CPU it happens to be exact; on GPU it is not, hence the tolerance.)

STOCK OOM IS THE POINT: the per-cell `compute_interacting_cell_scores` (and, at large
enough n_cells, the aggregates) will OOM in STOCK -- that is the bug the drop-ins fix. This
script CATCHES a stock OOM and reports it as "fix demonstrated" (then still runs the lowmem
version to confirm IT survives), rather than crashing. To actually check *equivalence* for a
function whose stock version OOMs at your scale, re-run with a smaller --M / --n-cells so
stock fits.

Manual smoke (real training/permutation, needs harreman + a GPU), kept out of pytest.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harreman  # noqa: E402  (real, installed on Savio)

from cell_communication_lowmem import (  # noqa: E402
    compute_cell_communication_lowmem,
    compute_ct_cell_communication_lowmem,
    compute_interacting_cell_scores_lowmem,
)

# parametric-path tolerance: absorbs GPU float64 reduction-order noise (~ULP), still ~1e6x
# tighter than anything a real algebraic bug would produce.
TOL_RTOL, TOL_ATOL = 1e-9, 1e-9


def _compare(name, a, b, kind):
    """kind='exact' -> must be bit-identical (non-parametric / gating path);
       kind='tol'   -> magnitude-aware (parametric path, GPU reduction noise allowed)."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        print(f"  [FAIL] {name}: shape {a.shape} != {b.shape}")
        return False
    with np.errstate(invalid="ignore"):
        diff = np.abs(a.astype(float) - b.astype(float))
    maxdiff = float(np.nanmax(diff)) if diff.size else 0.0
    if kind == "exact":
        ok = bool(np.array_equal(a, b)) or (
            bool(np.all(np.isnan(a) == np.isnan(b))) and (np.nanmax(diff) == 0.0)
        )
        note = "EXACT" if ok else "DIFF! (non-parametric path must be exact)"
    else:
        ok = bool(np.allclose(a, b, rtol=TOL_RTOL, atol=TOL_ATOL, equal_nan=True))
        note = "~ok (ULP)" if ok else "DIFF! (exceeds ULP tolerance)"
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: maxdiff={maxdiff:.3e} ({note})")
    return ok


# OOM exception type varies by torch version (torch.OutOfMemoryError / torch.cuda.OutOfMemoryError,
# both subclass RuntimeError); fall back to a message check.
_OOM_TYPES = tuple(t for t in (getattr(torch, "OutOfMemoryError", None),
                               getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None))
                   if isinstance(t, type))


def _is_oom(exc):
    if _OOM_TYPES and isinstance(exc, _OOM_TYPES):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _run_stock_guarded(fn, label, *args, **kwargs):
    """Run a stock harreman fn; if it OOMs, report that as the fix being demonstrated and
    return False (so the caller skips equivalence but still runs the lowmem version)."""
    try:
        fn(*args, **kwargs)
        return True
    except Exception as e:
        if not _is_oom(e):
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  [OOM] stock {label} ran out of GPU memory -- THIS IS THE BUG THE DROP-IN FIXES.")
        print(f"        ({str(e).splitlines()[0]})")
        print(f"        Running the lowmem version to confirm it survives; re-run with smaller "
              f"--M/--n-cells to check equivalence for {label}.")
        return False


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


def _setup_common(adata):
    harreman.pp.extract_interaction_db(adata, species="human", database="transporter", extracellular_only=True)
    harreman.tl.apply_gene_filtering(adata, layer_key="counts", model="danb", autocorrelation_filt=False)
    harreman.tl.compute_knn_graph(adata, compute_neighbors_on_key="spatial", n_neighbors=5, weighted_graph=False)
    harreman.tl.compute_gene_pairs(adata, ct_specific=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adata", required=True)
    ap.add_argument("--cell-type-col", required=True, help="obs column for the ct-aware test")
    ap.add_argument("--n-cells", type=int, default=8000)
    ap.add_argument("--M", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk-size", type=int, default=8, help="force small so chunking is exercised")
    # CU-E: the per-cell nbhd path (section 3) chunks TWO axes (gene-pair + metabolite),
    # independently of the aggregate functions' single --chunk-size. Default both to
    # --chunk-size so a plain invocation already exercises the chunked per-cell path on GPU.
    ap.add_argument("--nbhd-gp-chunk-size", type=int, default=None,
                    help="per-cell nbhd gene-pair chunk (default: --chunk-size)")
    ap.add_argument("--nbhd-m-chunk-size", type=int, default=None,
                    help="per-cell nbhd metabolite chunk (default: --chunk-size)")
    args = ap.parse_args()
    nbhd_gp_chunk = args.chunk_size if args.nbhd_gp_chunk_size is None else args.nbhd_gp_chunk_size
    nbhd_m_chunk = args.chunk_size if args.nbhd_m_chunk_size is None else args.nbhd_m_chunk_size

    print(f"device: {'cuda' if torch.cuda.is_available() else 'cpu'} | chunk_size={args.chunk_size} | M={args.M}")
    print("(non-parametric keys checked EXACT; parametric cs/Z checked ULP-tolerant -- see module docstring)")
    results = []  # (label, ok_or_None) ; None = stock OOM, equivalence not checked

    # ---- 1) cell-type-independent aggregate --------------------------------------------
    print("\n[1/3] compute_cell_communication vs _lowmem")
    a_stock = _load(args.adata, args.n_cells, args.seed); _setup_common(a_stock)
    a_low = a_stock.copy()
    cc_kw = dict(model="danb", M=args.M, test="both", layer_key_p_test="counts",
                 layer_key_np_test="log_norm", seed=args.seed)
    stock_ok = _run_stock_guarded(harreman.tl.compute_cell_communication, "compute_cell_communication", a_stock, **cc_kw)
    compute_cell_communication_lowmem(a_low, gene_pair_chunk_size=args.chunk_size, **cc_kw)
    if stock_ok:
        ok = True
        for grain in ("gp", "m"):
            ok &= _compare(f"p/{grain}/cs", a_stock.uns["ccc_results"]["p"][grain]["cs"], a_low.uns["ccc_results"]["p"][grain]["cs"], "tol")
            for k in ("pval", "FDR"):
                ok &= _compare(f"np/{grain}/{k}", a_stock.uns["ccc_results"]["np"][grain][k], a_low.uns["ccc_results"]["np"][grain][k], "exact")
            ok &= _compare(f"p/{grain}/Z", a_stock.uns["ccc_results"]["p"][grain]["Z"], a_low.uns["ccc_results"]["p"][grain]["Z"], "tol")
        results.append(("compute_cell_communication", ok))
    else:
        print("  [OK] lowmem compute_cell_communication completed without OOM.")
        results.append(("compute_cell_communication", None))

    # ---- 2) cell-type-aware aggregate --------------------------------------------------
    print("\n[2/3] compute_ct_cell_communication vs _lowmem")
    gp_filt = list(zip(a_stock.uns["ccc_results"]["cell_com_df_gp_sig"]["Gene 1"],
                       a_stock.uns["ccc_results"]["cell_com_df_gp_sig"]["Gene 2"])) \
        if stock_ok and "cell_com_df_gp_sig" in a_stock.uns.get("ccc_results", {}) else None
    b_stock = _load(args.adata, args.n_cells, args.seed); _setup_common(b_stock)
    for key in ("cell_type_pairs", "gene_pairs_per_ct_pair"):
        b_stock.uns.pop(key, None)
    harreman.tl.compute_gene_pairs(b_stock, cell_type_key=args.cell_type_col)
    b_low = b_stock.copy()
    ct_kw = dict(model="danb", cell_type_key=args.cell_type_col, M=args.M, test="both",
                 layer_key_p_test="counts", layer_key_np_test="log_norm",
                 subset_gene_pairs=gp_filt, fix_gp=False, seed=args.seed)
    stock_ct_ok = _run_stock_guarded(harreman.tl.compute_ct_cell_communication, "compute_ct_cell_communication", b_stock, **ct_kw)
    compute_ct_cell_communication_lowmem(b_low, gene_pair_chunk_size=args.chunk_size, **ct_kw)
    if stock_ct_ok:
        ok = True
        for grain in ("gp", "m"):
            ok &= _compare(f"p/{grain}/cs", b_stock.uns["ct_ccc_results"]["p"][grain]["cs"], b_low.uns["ct_ccc_results"]["p"][grain]["cs"], "tol")
            for k in ("pval", "FDR"):
                ok &= _compare(f"np/{grain}/{k}", b_stock.uns["ct_ccc_results"]["np"][grain][k], b_low.uns["ct_ccc_results"]["np"][grain][k], "exact")
            ok &= _compare(f"p/{grain}/Z", b_stock.uns["ct_ccc_results"]["p"][grain]["Z"], b_low.uns["ct_ccc_results"]["p"][grain]["Z"], "tol")
        results.append(("compute_ct_cell_communication", ok))
    else:
        print("  [OK] lowmem compute_ct_cell_communication completed without OOM.")
        results.append(("compute_ct_cell_communication", None))

    # ---- 3) per-cell interacting-cell scores (nbhd path) -------------------------------
    print("\n[3/3] compute_interacting_cell_scores vs _lowmem")
    if not stock_ok:
        print("  [SKIP] need the cell-indep ccc_results (stock OOM'd above); re-run smaller.")
        results.append(("compute_interacting_cell_scores", None))
    else:
        harreman.tl.select_significant_interactions(a_stock, test="non-parametric", threshold=0.05)
        c_stock = a_stock.copy(); c_low = a_stock.copy()
        ics_kw = dict(test="non-parametric", restrict_significance="both",
                      compute_significance="non-parametric", M=args.M, seed=args.seed)
        stock_ics_ok = _run_stock_guarded(harreman.tl.compute_interacting_cell_scores, "compute_interacting_cell_scores", c_stock, **ics_kw)
        # CU-E: force small chunks on BOTH nbhd axes so the two-pass chunked per-cell path is
        # actually exercised on GPU (else it runs one adaptive chunk and the memory fix is
        # untested). Non-parametric cs/pval/FDR must still be EXACT vs stock (both grains).
        print(f"  [nbhd] gene_pair_chunk_size={nbhd_gp_chunk} metabolite_chunk_size={nbhd_m_chunk}")
        compute_interacting_cell_scores_lowmem(
            c_low, gene_pair_chunk_size=nbhd_gp_chunk, metabolite_chunk_size=nbhd_m_chunk, **ics_kw
        )
        if stock_ics_ok:
            ok = True
            for grain in ("gp", "m"):
                for k in ("cs", "pval", "FDR"):
                    ok &= _compare(f"np/{grain}/{k}", c_stock.uns["interacting_cell_results"]["np"][grain][k],
                                   c_low.uns["interacting_cell_results"]["np"][grain][k], "exact")
            results.append(("compute_interacting_cell_scores", ok))
        else:
            print("  [OK] lowmem compute_interacting_cell_scores completed without OOM.")
            results.append(("compute_interacting_cell_scores", None))

    # ---- verdict -----------------------------------------------------------------------
    print("\n=== summary ===")
    real_fail = False
    for label, ok in results:
        if ok is None:
            print(f"  {label}: stock OOM -> fix demonstrated, equivalence not checked (re-run smaller)")
        elif ok:
            print(f"  {label}: EQUIVALENT (non-parametric exact, parametric within ULP)")
        else:
            print(f"  {label}: MISMATCH -- inspect DIFF rows above")
            real_fail = True
    print("\n" + ("REAL MISMATCH DETECTED ❌" if real_fail else "OK ✅ (no non-parametric divergence; parametric within ULP)"))
    sys.exit(1 if real_fail else 0)


if __name__ == "__main__":
    main()
