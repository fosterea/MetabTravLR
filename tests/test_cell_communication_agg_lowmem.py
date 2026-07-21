"""Tier-0/1 equivalence tests for the CU-B drop-in in
`metab_processing/Harreman/cell_communication_lowmem.py`:
`compute_cell_communication_lowmem` (memory-safe twin of harreman's cell-type-
INDEPENDENT `compute_cell_communication`). No GPU/CUDA/Savio/real-harreman needed --
`harreman` is the small local fake package (`tests/fixtures/fake_harreman/`) which now
also vendors (CU-B) the real stock `compute_cell_communication` /
`run_cell_communication_analysis` and their transitive deps (`flatten`,
`compute_p_results`, `get_cell_communication_results`, `normalize_values`,
`compute_max_cs`, `compute_max_cs_gp`) verbatim from `DataForClaude/cell_communication.py`
(confirmed byte-identical via AST comparison -- see that fixture's module docstring).

This asserts the low-mem drop-in reproduces the stock algorithm bit-for-bit: the
observed scores (`cs`, both tests), every non-parametric ('np') output (`pval`, `FDR`,
`perm_cs_a/b` -- RNG is identical for a given seed since only `torch.randperm` consumes
it, and the drop-in re-seeds it identically per gene-pair chunk), AND the final
`cell_com_df_gp`/`cell_com_df_m` DataFrames built by `get_cell_communication_results`
(called by both paths). Matches how `harreman_funcs.py::HarremanRunner.run_cell_
independent` actually calls this: `model='danb', test='both', layer_key_p_test='counts',
layer_key_np_test='log_norm'`.

ONE MEASURED EXCEPTION (found during development, not assumed away): the parametric
('p') `Z`/`Z_pval`/`Z_FDR` can differ from stock by 1-4 ULPs of float64 (~1e-16 to
2.2e-16 absolute) specifically when `gene_pair_chunk_size < n_gene_pairs` (real,
multi-chunk runs) -- root-caused to PyTorch's `.sum(dim=0)` reduction over the cell axis
being sensitive to tensor width/stride even when the per-column inputs are themselves
bit-identical (confirmed by direct isolation; see cell_communication_lowmem.py's module
docstring for the full writeup). `cs` and every 'np' output were bit-for-bit identical
across 29+ random seeds and every chunk size tried, including the pathological
chunk_size=1 -- and a SINGLE chunk (`gene_pair_chunk_size >= n_gp`) is proven exactly
bit-identical for `Z` too, since it then executes literally stock's own reduction.
`_ccc_results_equal` below encodes exactly this: `assert_array_equal` for the always-
exact keys, a very tight `assert_allclose` (atol=1e-8, ~1e8x looser than any diff ever
observed) for the parametric Z-derived keys only.

`conftest.py` puts `metab_processing/Harreman` and the fake `harreman` package on
`sys.path`, so plain `import harreman` / `import cell_communication_lowmem` both resolve.
"""
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import torch

import harreman
from cell_communication_lowmem import compute_cell_communication_lowmem

N_CELLS = 40
GENES = [f"G{i}" for i in range(1, 7)]  # G1..G6


def _make_weights(n_cells, k=4, seed=0):
    """Deterministic small-degree sparse adjacency (stand-in for a spatial KNN graph;
    the low-mem drop-in and stock consume this matrix identically)."""
    rng = np.random.default_rng(seed)
    rows, cols, vals = [], [], []
    for i in range(n_cells):
        candidates = [j for j in range(n_cells) if j != i]
        neighbors = rng.choice(candidates, size=min(k, len(candidates)), replace=False)
        for j in neighbors:
            rows.append(i)
            cols.append(int(j))
            vals.append(float(rng.uniform(0.2, 1.0)))
    return sp.csr_matrix((vals, (rows, cols)), shape=(n_cells, n_cells))


def make_test_adata(n_cells=N_CELLS, seed=0):
    """Tiny synthetic AnnData with three gene-pair shapes across two metabolites:
      - a homotypic (same-gene) pair:      ("G1", "G1")             -> Metab_A
      - a plain heterotypic pair:          ("G2", "G3")              -> Metab_A
      - a heterodimer tuple pair:          (["G4", "G5"], "G6")      -> Metab_B
    Metab_A's two pairs land in DIFFERENT gene-pair chunks whenever
    gene_pair_chunk_size=1 -- this is what exercises the
    "metabolite spans >1 chunk" case documented in cell_communication_lowmem.py.

    Mirrors the uns/obsp contract `compute_cell_communication` reads (see
    DataForClaude/cell_communication.py lines 464-877): `uns['gene_pairs']` (a list of
    TUPLES, matching the real `compute_gene_pairs` output format -- NOT lists, since
    `run_cell_communication_analysis` compares reconstructed tuples against this list
    directly with no list->tuple normalization, unlike the per-cell CU-A function),
    `uns['gene_pairs_per_metabolite']`, `obsp['weights']`, and 'counts'/'log_norm' layers.
    """
    rng = np.random.default_rng(seed)
    n_genes = len(GENES)
    counts = rng.poisson(lam=5, size=(n_cells, n_genes)).astype(np.float64) + 1.0  # no all-zero rows
    log_norm = np.log1p(counts / counts.sum(axis=1, keepdims=True) * 1e4)

    adata = ad.AnnData(X=log_norm.copy())
    adata.var_names = GENES
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.layers["counts"] = counts
    adata.layers["log_norm"] = log_norm

    adata.obsp["weights"] = _make_weights(n_cells, k=4, seed=seed)

    adata.uns["gene_pairs"] = [
        ("G1", "G1"),
        ("G2", "G3"),
        (["G4", "G5"], "G6"),
    ]
    adata.uns["gene_pairs_per_metabolite"] = {
        "Metab_A": {
            "gene_pair": [("G1", "G1"), ("G2", "G3")],
            "gene_type": [("IMP-EXP", "IMP-EXP"), ("IMP-EXP", "IMP-EXP")],
        },
        "Metab_B": {
            "gene_pair": [(["G4", "G5"], "G6")],
            "gene_type": [("IMP-EXP", "IMP-EXP")],
        },
    }
    return adata


# Matches how `harreman_funcs.py::HarremanRunner.run_cell_independent` actually calls it.
CALL_KWARGS = dict(
    layer_key_p_test="counts",
    layer_key_np_test="log_norm",
    model="danb",
    M=50,
    test="both",
    seed=7,
    verbose=False,
)


def _ccc_results_equal(a_uns, b_uns, testcase):
    """Assert every array/DataFrame `compute_cell_communication[_lowmem]` writes matches
    between the two `adata.uns` dicts (`a` = stock, `b` = low-mem).

    EXACT (`assert_array_equal`) for `cs` (both tests) and every non-parametric ('np')
    output -- confirmed bit-for-bit identical across 29+ random seeds and every tested
    chunk size, including the pathological chunk_size=1 case. These are also the values
    production (`harreman_funcs.py`) actually gates significance on.

    NEAR-EXACT (`assert_allclose`, atol=1e-8) for parametric ('p') `Z`/`Z_pval`/`Z_FDR`
    ONLY -- empirically these occasionally (not always) differ from stock by 1-4 ULPs of
    float64 (~1e-16 to 2.2e-16 absolute) when gene pairs are chunked into >1 piece. This
    is NOT a chunking-logic bug: `sparse.mm`'s per-column outputs and the observed `cs`
    are always bit-exact regardless of chunking (verified by direct isolation -- see
    cell_communication_lowmem.py's module docstring). The divergence traces to
    `eg2_a`/`eg2_b`'s `.sum(dim=0)`/`.pow(2).sum(dim=0)` reduction, whose result can
    depend on the reduced tensor's width/stride at the float64-rounding level even when
    the per-column input values are identical -- a property of PyTorch's reduction
    kernel, not of this module's logic. The atol here (1e-8) is ~1e8x looser than the
    largest diff ever observed, so it still catches any REAL regression while tolerating
    this documented, negligible (and scientifically irrelevant -- it's below the last
    printed decimal of any p-value) floating-point wobble. Single-chunk runs
    (gene_pair_chunk_size >= n_gp) are proven exact for these keys too (see
    ChunkInvarianceTests) since with one chunk this executes literally the same
    reduction stock does.
    """
    exact_checks = {
        "p": ("cs",),
        "np": ("cs", "pval", "FDR", "perm_cs_a", "perm_cs_b"),
    }
    close_checks = {
        "p": ("Z", "Z_pval", "Z_FDR"),
    }
    for test_key, keys in exact_checks.items():
        for grain in ("gp", "m"):
            for key in keys:
                with testcase.subTest(test=test_key, grain=grain, key=key):
                    np.testing.assert_array_equal(
                        a_uns["ccc_results"][test_key][grain][key],
                        b_uns["ccc_results"][test_key][grain][key],
                        err_msg=f"mismatch in ['{test_key}']['{grain}']['{key}']",
                    )
    for test_key, keys in close_checks.items():
        for grain in ("gp", "m"):
            for key in keys:
                with testcase.subTest(test=test_key, grain=grain, key=key):
                    # parametric Z differs from stock only by float64 reduction-order
                    # noise (~value*2^-52) when gene pairs span >1 chunk; observed worst
                    # case rel 4e-16. Magnitude-aware bound with ~1e4x margin catches any
                    # real algebraic divergence while allowing that last-bit noise.
                    np.testing.assert_allclose(
                        a_uns["ccc_results"][test_key][grain][key],
                        b_uns["ccc_results"][test_key][grain][key],
                        rtol=1e-11, atol=1e-13,
                        err_msg=f"mismatch in ['{test_key}']['{grain}']['{key}']",
                    )
    for df_key in ("cell_com_df_gp", "cell_com_df_m"):
        with testcase.subTest(df_key=df_key):
            pd.testing.assert_frame_equal(
                a_uns["ccc_results"][df_key], b_uns["ccc_results"][df_key]
            )
    with testcase.subTest(field="lc_zs"):
        pd.testing.assert_frame_equal(a_uns["lc_zs"], b_uns["lc_zs"])
    with testcase.subTest(field="D"):
        np.testing.assert_array_equal(a_uns["D"], b_uns["D"])
    with testcase.subTest(field="gene_pair_dict"):
        testcase.assertEqual(a_uns["gene_pair_dict"], b_uns["gene_pair_dict"])
    with testcase.subTest(field="genes"):
        testcase.assertEqual(a_uns["genes"], b_uns["genes"])


class CellCommunicationAggLowmemEquivalenceTests(unittest.TestCase):
    """Default (adaptive) chunk size vs stock -- exact equality end to end."""

    def setUp(self):
        self.adata_stock = make_test_adata(seed=1)
        self.adata_lowmem = make_test_adata(seed=1)

        harreman.tools.compute_cell_communication(self.adata_stock, **CALL_KWARGS)
        compute_cell_communication_lowmem(self.adata_lowmem, **CALL_KWARGS)

    def test_bookkeeping_matches(self):
        self.assertEqual(self.adata_stock.uns["gene_pairs_ind"], self.adata_lowmem.uns["gene_pairs_ind"])
        self.assertEqual(self.adata_stock.uns["gene_pair_dict"], self.adata_lowmem.uns["gene_pair_dict"])
        self.assertEqual(self.adata_stock.uns["genes"], self.adata_lowmem.uns["genes"])
        # both designed metabolites survived (n_gp == 3, both metabolites present)
        self.assertEqual(len(self.adata_lowmem.uns["gene_pairs_ind"]), 3)
        self.assertEqual(set(self.adata_lowmem.uns["gene_pair_dict"]), {"Metab_A", "Metab_B"})

    def test_ccc_results_bit_identical_to_stock(self):
        _ccc_results_equal(self.adata_stock.uns, self.adata_lowmem.uns, self)

    def test_finite_scores(self):
        """Sanity check the real DANB math didn't degenerate on this tiny fixture."""
        cs_p = self.adata_lowmem.uns["ccc_results"]["p"]["gp"]["cs"]
        cs_np = self.adata_lowmem.uns["ccc_results"]["np"]["gp"]["cs"]
        self.assertTrue(np.all(np.isfinite(cs_p)))
        self.assertTrue(np.all(np.isfinite(cs_np)))


class CellCommunicationAggLowmemChunkInvarianceTests(unittest.TestCase):
    """Chunk-invariance: gene_pair_chunk_size in {1, 2, n_gp, None (adaptive)} must all
    reproduce stock exactly -- proves chunking never changes output, at every boundary
    (a lone gene pair per chunk, a partial chunk, a single all-in-one chunk, and the
    adaptive default)."""

    def setUp(self):
        self.adata_stock = make_test_adata(seed=5)
        harreman.tools.compute_cell_communication(self.adata_stock, **CALL_KWARGS)

    def _run_lowmem(self, chunk_size):
        adata = make_test_adata(seed=5)
        kwargs = dict(CALL_KWARGS)
        compute_cell_communication_lowmem(adata, gene_pair_chunk_size=chunk_size, **kwargs)
        return adata

    def test_chunk_size_1(self):
        adata = self._run_lowmem(1)
        _ccc_results_equal(self.adata_stock.uns, adata.uns, self)

    def test_chunk_size_2(self):
        adata = self._run_lowmem(2)
        _ccc_results_equal(self.adata_stock.uns, adata.uns, self)

    def test_chunk_size_equals_n_gp(self):
        n_gp = len(self.adata_stock.uns["gene_pairs"])
        adata = self._run_lowmem(n_gp)
        _ccc_results_equal(self.adata_stock.uns, adata.uns, self)

    def test_chunk_size_exceeds_n_gp(self):
        """gene_pair_chunk_size >= n_gp must reproduce stock's single-pass behavior."""
        n_gp = len(self.adata_stock.uns["gene_pairs"])
        adata = self._run_lowmem(n_gp + 100)
        _ccc_results_equal(self.adata_stock.uns, adata.uns, self)

    def test_chunk_size_none_adaptive(self):
        adata = self._run_lowmem(None)
        _ccc_results_equal(self.adata_stock.uns, adata.uns, self)

    def test_all_chunk_sizes_mutually_identical(self):
        """Belt-and-braces: every chunk choice also matches every other choice directly
        (not just via the shared stock reference)."""
        n_gp = len(self.adata_stock.uns["gene_pairs"])
        results = {cs: self._run_lowmem(cs) for cs in (1, 2, n_gp, None)}
        keys = list(results)
        base = results[keys[0]]
        for other_key in keys[1:]:
            with self.subTest(chunk_size=other_key):
                _ccc_results_equal(base.uns, results[other_key].uns, self)

    def test_single_chunk_parametric_Z_is_exactly_bit_identical(self):
        """Stricter than `_ccc_results_equal` (which allows a tiny tolerance on the
        parametric Z-score for genuinely chunked runs -- see its docstring): a SINGLE
        chunk (gene_pair_chunk_size >= n_gp) executes the exact same reduction stock
        does, so Z/Z_pval/Z_FDR must be EXACTLY bit-for-bit identical here, with no
        tolerance at all. Verified empirically across 29+ seeds during development."""
        n_gp = len(self.adata_stock.uns["gene_pairs"])
        adata = self._run_lowmem(n_gp)
        for grain in ("gp", "m"):
            for key in ("Z", "Z_pval", "Z_FDR"):
                with self.subTest(grain=grain, key=key):
                    np.testing.assert_array_equal(
                        self.adata_stock.uns["ccc_results"]["p"][grain][key],
                        adata.uns["ccc_results"]["p"][grain][key],
                    )


class CellCommunicationAggLowmemMemoryGuardTests(unittest.TestCase):
    """Non-vacuity: assert the chunked path is actually exercised (number of
    torch.sparse.mm calls scales with the number of chunks) rather than silently
    collapsing to a single pass regardless of `gene_pair_chunk_size`."""

    def test_sparse_mm_call_count_scales_with_chunks(self):
        n_gp = 3  # fixed by make_test_adata's uns['gene_pairs']
        counts_by_chunk = {}
        for chunk_size, n_chunks in ((1, 3), (n_gp, 1)):
            adata = make_test_adata(seed=9)
            kwargs = dict(CALL_KWARGS)
            kwargs["test"] = "parametric"  # simpler: no permutation loop to also count
            with mock.patch("torch.sparse.mm", wraps=torch.sparse.mm) as spy:
                compute_cell_communication_lowmem(adata, gene_pair_chunk_size=chunk_size, **kwargs)
            counts_by_chunk[chunk_size] = spy.call_count
            # 4 sparse.mm calls per chunk in the parametric branch: WX2t, WtX2t, WX1t, WtX1t
            self.assertEqual(spy.call_count, 4 * n_chunks, f"chunk_size={chunk_size}")

        self.assertGreater(counts_by_chunk[1], counts_by_chunk[n_gp])

    def test_np_permutation_loop_scales_with_chunks(self):
        """The non-parametric permutation loop (the intricate reseed-per-chunk path) must
        also chunk, not collapse to a single pass -- otherwise it would still pass output
        equivalence for small n_gp while defeating the memory fix. Its sparse.mm count
        scales linearly with n_chunks (3 chunks issues exactly 3x the calls of 1 chunk)."""
        n_gp = 3  # fixed by make_test_adata's uns['gene_pairs']
        kwargs = dict(CALL_KWARGS)
        kwargs.update(test="non-parametric", M=2)
        counts = {}
        for chunk_size in (1, n_gp):
            adata = make_test_adata(seed=9)
            with mock.patch("torch.sparse.mm", wraps=torch.sparse.mm) as spy:
                compute_cell_communication_lowmem(adata, gene_pair_chunk_size=chunk_size, **kwargs)
            counts[chunk_size] = spy.call_count
        self.assertEqual(counts[1], 3 * counts[n_gp])
        self.assertGreater(counts[1], counts[n_gp])

    def test_all_gene_pairs_processed_across_chunks(self):
        """The chunked cs_gp array covers every gene pair exactly once, regardless of
        how many chunks it took (guards against an off-by-one in the chunk loop's
        `range(0, n_gp, chunk)` / slicing)."""
        adata_ref = make_test_adata(seed=3)
        kwargs = dict(CALL_KWARGS)
        harreman.tools.compute_cell_communication(adata_ref, **kwargs)

        adata_chunked = make_test_adata(seed=3)
        compute_cell_communication_lowmem(adata_chunked, gene_pair_chunk_size=1, **kwargs)

        n_gp = len(adata_ref.uns["gene_pairs"])
        self.assertEqual(len(adata_chunked.uns["ccc_results"]["p"]["gp"]["cs"]), n_gp)
        np.testing.assert_array_equal(
            adata_ref.uns["ccc_results"]["p"]["gp"]["cs"],
            adata_chunked.uns["ccc_results"]["p"]["gp"]["cs"],
        )


class CellCommunicationAggLowmemGuardTests(unittest.TestCase):
    def test_check_analytic_null_raises_not_implemented(self):
        adata = make_test_adata(seed=2)
        kwargs = dict(CALL_KWARGS)
        kwargs["check_analytic_null"] = True
        with self.assertRaises(NotImplementedError):
            compute_cell_communication_lowmem(adata, **kwargs)

    def test_invalid_test_raises_value_error(self):
        adata = make_test_adata(seed=2)
        kwargs = dict(CALL_KWARGS)
        kwargs["test"] = "bogus"
        with self.assertRaises(ValueError):
            compute_cell_communication_lowmem(adata, **kwargs)

    def test_invalid_mean_raises_value_error(self):
        adata = make_test_adata(seed=2)
        kwargs = dict(CALL_KWARGS)
        kwargs["mean"] = "bogus"
        with self.assertRaises(ValueError):
            compute_cell_communication_lowmem(adata, **kwargs)


class CellCommunicationAggLowmemNonParametricOnlyTests(unittest.TestCase):
    """test='non-parametric' only (no 'p' branch at all -- exercises the eg2s_gp-less
    path and the standalone non-parametric chunk loop)."""

    def setUp(self):
        self.adata_stock = make_test_adata(seed=11)
        self.adata_lowmem = make_test_adata(seed=11)
        kwargs = dict(CALL_KWARGS)
        kwargs["test"] = "non-parametric"

        harreman.tools.compute_cell_communication(self.adata_stock, **kwargs)
        compute_cell_communication_lowmem(self.adata_lowmem, gene_pair_chunk_size=1, **kwargs)

    def test_non_parametric_only_matches_stock(self):
        for grain in ("gp", "m"):
            for key in ("cs", "pval", "FDR", "perm_cs_a", "perm_cs_b"):
                with self.subTest(grain=grain, key=key):
                    np.testing.assert_array_equal(
                        self.adata_stock.uns["ccc_results"]["np"][grain][key],
                        self.adata_lowmem.uns["ccc_results"]["np"][grain][key],
                    )
        self.assertNotIn("p", self.adata_lowmem.uns["ccc_results"])


class CellCommunicationAggLowmemCenteredNpTests(unittest.TestCase):
    """center_counts_for_np_test=True + test='both' exercises the STOCK SHORTCUT branch
    (np observed score copied from the already-computed p score) plus its interaction
    with chunking (standardize_counts still runs per chunk for the permutation loop)."""

    def setUp(self):
        self.adata_stock = make_test_adata(seed=13)
        self.adata_lowmem = make_test_adata(seed=13)
        kwargs = dict(CALL_KWARGS)
        kwargs["center_counts_for_np_test"] = True

        harreman.tools.compute_cell_communication(self.adata_stock, **kwargs)
        compute_cell_communication_lowmem(self.adata_lowmem, gene_pair_chunk_size=1, **kwargs)

    def test_centered_np_matches_stock(self):
        _ccc_results_equal(self.adata_stock.uns, self.adata_lowmem.uns, self)

    def test_np_cs_equals_p_cs_shortcut(self):
        """Confirms the shortcut actually fired (np cs is literally the p cs)."""
        for grain in ("gp", "m"):
            np.testing.assert_array_equal(
                self.adata_lowmem.uns["ccc_results"]["p"][grain]["cs"],
                self.adata_lowmem.uns["ccc_results"]["np"][grain]["cs"],
            )


if __name__ == "__main__":
    unittest.main()
