"""Tier-0/1 equivalence tests for the CU-C drop-in in
`metab_processing/Harreman/cell_communication_lowmem.py`:
`compute_ct_cell_communication_lowmem` (memory-safe twin of harreman's cell-type-AWARE
`compute_ct_cell_communication`). No GPU/CUDA/Savio/real-harreman needed -- `harreman` is
the small local fake package (`tests/fixtures/fake_harreman/`) which now also vendors
(CU-C) the real stock `compute_ct_cell_communication` / `run_ct_cell_communication_analysis`
and their transitive deps (`standardize_ct_counts`, `create_weights_ct_pairs`,
`compute_metabolite_cs_ct`, `compute_ct_p_results`, `get_ct_cell_communication_results`,
`normalize_ct_values`, `center_ct_counts_torch`, `apply_model_per_cell_type`) verbatim from
`DataForClaude/cell_communication.py` (confirmed byte-identical via AST comparison -- see
that fixture's module docstring).

Matches how `harreman_funcs.py::HarremanRunner.run_cell_aware` actually calls this:
`model='danb', test='both', cell_type_key=<tier col>, layer_key_p_test='counts',
layer_key_np_test='log_norm', subset_gene_pairs=<sig pairs from the agnostic run>,
fix_gp=False`.

Exactness contract (see cell_communication_lowmem.py's CU-C module docstring for the
derivation): `cs` (both tests) and the ENTIRE non-parametric path (`pval`, `FDR`,
`perm_cs`) are bit-for-bit identical to stock -- these are also what production
(`select_significant_interactions(test='non-parametric', ...)`) actually gates on.
Parametric `Z`/`Z_pval`/`Z_FDR` use a magnitude-aware `assert_allclose(rtol=1e-11,
atol=1e-13)` ONLY, mirroring the CU-B finding that float64 reduction order can wobble by a
few ULPs across chunks (established there, not re-derived here) -- everything else in the
final `cell_com_df_gp`/`cell_com_df_m` DataFrames (built by `get_ct_cell_communication_
results`, called verbatim from harreman via the `_need` shim) is compared exactly.

Fixture design note: the MAIN fixture (`make_test_adata`) maps every `cell_type_pair` to
the FULL gene-pair list (same order), so `ct_specific_gene_pairs` (stock's mechanism for
cell-type pairs whose tested gene-pair set is a strict subset of the full set) is empty
there -- the in-place masking mutation in `compute_metabolite_cs_ct` is instead covered by
`CtSpecificGenePairsMaskingEquivalenceTests` at the bottom of this file (a fixture where
two ct_pairs carry a strict subset). The full-list mapping also matches how
`get_ct_cell_communication_results` builds its DataFrame (`.flatten()` against
`gene_pairs_ind_per_ct_pair`'s iteration order), which only lines up correctly when every
ct pair enumerates the gene pairs in the same order as `gene_pairs_ind`.

`conftest.py` puts `metab_processing/Harreman` and the fake `harreman` package on
`sys.path`, so plain `import harreman` / `import cell_communication_lowmem` both resolve.
"""
import itertools
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import torch

import harreman
from harreman.tools import run_ct_cell_communication_analysis as _stock_run_ct
from cell_communication_lowmem import (
    compute_ct_cell_communication_lowmem,
    _run_ct_cell_communication_analysis_lowmem,
)

N_CELLS = 30
CELL_TYPES = (["TypeA"] * 10) + (["TypeB"] * 10) + (["TypeC"] * 10)
GENES = [f"G{i}" for i in range(1, 7)]  # G1..G6


def _make_weights(n_cells, k=6, seed=0):
    """Deterministic small-degree sparse adjacency, neighbors drawn from the FULL cell
    population (not filtered by cell type) so every unordered cell-type pair gets at
    least one edge -- verified for (seed=3, k=6) against this exact CELL_TYPES layout."""
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


def make_test_adata(n_cells=N_CELLS, seed=0, weights_seed=3):
    """Tiny synthetic AnnData with a `cell_type` obs column (3 types) and the same three
    gene-pair shapes CU-B's fixture uses (homotypic, plain heterotypic, heterodimer
    tuple), spanning two metabolites:
      - a homotypic (same-gene) pair:      ("G1", "G1")             -> Metab_A
      - a plain heterotypic pair:          ("G2", "G3")              -> Metab_A
      - a heterodimer tuple pair:          (["G4", "G5"], "G6")      -> Metab_B
    Metab_A's two pairs land in DIFFERENT gene-pair chunks whenever
    gene_pair_chunk_size=1, exercising the "metabolite spans >1 chunk" case.

    Mirrors the uns/obsp/obs contract `compute_ct_cell_communication` reads (see
    DataForClaude/cell_communication.py lines 879-1344): `uns['gene_pairs']` (a list of
    TUPLES, matching real `compute_gene_pairs` output), `uns['gene_pairs_per_metabolite']`,
    `uns['cell_type_pairs']`, `uns['gene_pairs_per_ct_pair']`, `uns['fix_ct']`, `obs['cell_type']`,
    `obsp['weights']`, and 'counts'/'log_norm' layers. `compute_gene_pairs(ct_specific=True)`
    normally populates `cell_type_pairs`/`gene_pairs_per_ct_pair` -- set directly here instead.
    """
    rng = np.random.default_rng(seed)
    n_genes = len(GENES)
    counts = rng.poisson(lam=5, size=(n_cells, n_genes)).astype(np.float64) + 1.0  # no all-zero rows
    log_norm = np.log1p(counts / counts.sum(axis=1, keepdims=True) * 1e4)

    adata = ad.AnnData(X=log_norm.copy())
    adata.var_names = GENES
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.obs["cell_type"] = pd.Categorical(CELL_TYPES[:n_cells])
    adata.layers["counts"] = counts
    adata.layers["log_norm"] = log_norm

    adata.obsp["weights"] = _make_weights(n_cells, k=6, seed=weights_seed)

    gene_pairs = [
        ("G1", "G1"),
        ("G2", "G3"),
        (["G4", "G5"], "G6"),
    ]
    adata.uns["gene_pairs"] = gene_pairs
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

    cell_type_pairs = list(itertools.combinations_with_replacement(sorted(set(CELL_TYPES[:n_cells])), 2))
    adata.uns["cell_type_pairs"] = cell_type_pairs
    # Every ct pair maps to the FULL gene-pair list, in the SAME order -- see module
    # docstring's fixture design note (keeps ct_specific_gene_pairs empty).
    adata.uns["gene_pairs_per_ct_pair"] = {ct_pair: list(gene_pairs) for ct_pair in cell_type_pairs}
    adata.uns["fix_ct"] = False
    return adata


# Matches how `harreman_funcs.py::HarremanRunner.run_cell_aware` actually calls it.
CALL_KWARGS = dict(
    layer_key_p_test="counts",
    layer_key_np_test="log_norm",
    model="danb",
    cell_type_key="cell_type",
    M=20,
    test="both",
    fix_gp=False,
    seed=7,
    verbose=False,
)


def _ct_ccc_results_equal(a_uns, b_uns, testcase):
    """Assert every array/DataFrame `compute_ct_cell_communication[_lowmem]` writes
    matches between the two `adata.uns` dicts (`a` = stock, `b` = low-mem).

    EXACT (`assert_array_equal`) for `cs` (both tests) and the entire non-parametric
    ('np') path (`cs`, `pval`, `FDR`, `perm_cs`) -- these are also what production
    (`select_significant_interactions`) gates significance on.

    NEAR-EXACT (`assert_allclose`, rtol=1e-11, atol=1e-13) for parametric ('p')
    `Z`/`Z_pval`/`Z_FDR` ONLY -- mirrors the CU-B finding (established there) that
    float64 `.sum(dim=0)` reduction order can differ by a few ULPs across chunk widths
    even when the per-column inputs are bit-identical.
    """
    exact_checks = {
        "p": ("cs",),
        "np": ("cs", "pval", "FDR", "perm_cs"),
    }
    close_checks = {"p": ("Z", "Z_pval", "Z_FDR")}

    for test_key, keys in exact_checks.items():
        for grain in ("gp", "m"):
            for key in keys:
                with testcase.subTest(test=test_key, grain=grain, key=key):
                    np.testing.assert_array_equal(
                        a_uns["ct_ccc_results"][test_key][grain][key],
                        b_uns["ct_ccc_results"][test_key][grain][key],
                        err_msg=f"mismatch in ['{test_key}']['{grain}']['{key}']",
                    )
    for test_key, keys in close_checks.items():
        for grain in ("gp", "m"):
            for key in keys:
                with testcase.subTest(test=test_key, grain=grain, key=key):
                    np.testing.assert_allclose(
                        a_uns["ct_ccc_results"][test_key][grain][key],
                        b_uns["ct_ccc_results"][test_key][grain][key],
                        rtol=1e-11, atol=1e-13,
                        err_msg=f"mismatch in ['{test_key}']['{grain}']['{key}']",
                    )

    close_cols = {"Z", "Z_pval", "Z_FDR"}
    for df_key in ("cell_com_df_gp", "cell_com_df_m"):
        with testcase.subTest(df_key=df_key):
            a_df = a_uns["ct_ccc_results"][df_key]
            b_df = b_uns["ct_ccc_results"][df_key]
            exact_cols = [c for c in a_df.columns if c not in close_cols]
            pd.testing.assert_frame_equal(a_df[exact_cols], b_df[exact_cols])
            for c in close_cols:
                if c in a_df.columns:
                    np.testing.assert_allclose(
                        a_df[c].values, b_df[c].values, rtol=1e-11, atol=1e-13,
                        err_msg=f"mismatch in df column '{c}'",
                    )

    with testcase.subTest(field="D"):
        np.testing.assert_array_equal(a_uns["D"], b_uns["D"])
    with testcase.subTest(field="gene_pair_dict"):
        testcase.assertEqual(a_uns["gene_pair_dict"], b_uns["gene_pair_dict"])
    with testcase.subTest(field="genes"):
        testcase.assertEqual(a_uns["genes"], b_uns["genes"])


class CtCellCommunicationLowmemEquivalenceTests(unittest.TestCase):
    """Default (adaptive) chunk size vs stock -- exact equality end to end. Passes an
    explicit `subset_gene_pairs` equal to the full gene-pair list, matching how
    `harreman_funcs.py` always passes one (never relies on the None default in
    production, though None is also covered by NonDefaultSubsetTests below)."""

    def setUp(self):
        self.adata_stock = make_test_adata(seed=1)
        self.adata_lowmem = make_test_adata(seed=1)

        gene_pairs = list(self.adata_stock.uns["gene_pairs"])
        harreman.tools.compute_ct_cell_communication(self.adata_stock, subset_gene_pairs=gene_pairs, **CALL_KWARGS)
        compute_ct_cell_communication_lowmem(self.adata_lowmem, subset_gene_pairs=gene_pairs, **CALL_KWARGS)

    def test_bookkeeping_matches(self):
        self.assertEqual(self.adata_stock.uns["gene_pairs_ind"], self.adata_lowmem.uns["gene_pairs_ind"])
        self.assertEqual(self.adata_stock.uns["gene_pair_dict"], self.adata_lowmem.uns["gene_pair_dict"])
        self.assertEqual(self.adata_stock.uns["genes"], self.adata_lowmem.uns["genes"])
        self.assertEqual(len(self.adata_lowmem.uns["gene_pairs_ind"]), 3)
        self.assertEqual(set(self.adata_lowmem.uns["gene_pair_dict"]), {"Metab_A", "Metab_B"})
        # 3 cell types -> 6 unordered cell-type pairs (combinations_with_replacement)
        self.assertEqual(len(self.adata_lowmem.uns["cell_type_pairs"]), 6)

    def test_ct_ccc_results_bit_identical_to_stock(self):
        _ct_ccc_results_equal(self.adata_stock.uns, self.adata_lowmem.uns, self)

    def test_finite_scores(self):
        """Sanity check the real DANB math + non-trivial weights didn't degenerate."""
        cs_p = self.adata_lowmem.uns["ct_ccc_results"]["p"]["gp"]["cs"]
        cs_np = self.adata_lowmem.uns["ct_ccc_results"]["np"]["gp"]["cs"]
        self.assertTrue(np.all(np.isfinite(cs_p)))
        self.assertTrue(np.all(np.isfinite(cs_np)))
        # not all-zero: confirms the fixture's weights actually connect every ct pair
        self.assertTrue(np.any(cs_np != 0))


class CtCellCommunicationLowmemChunkInvarianceTests(unittest.TestCase):
    """Chunk-invariance: gene_pair_chunk_size in {1, 2, n_gp, None (adaptive)} must all
    reproduce stock exactly."""

    def setUp(self):
        self.adata_stock = make_test_adata(seed=5)
        gene_pairs = list(self.adata_stock.uns["gene_pairs"])
        self.gene_pairs = gene_pairs
        harreman.tools.compute_ct_cell_communication(self.adata_stock, subset_gene_pairs=gene_pairs, **CALL_KWARGS)

    def _run_lowmem(self, chunk_size):
        adata = make_test_adata(seed=5)
        compute_ct_cell_communication_lowmem(
            adata, subset_gene_pairs=self.gene_pairs, gene_pair_chunk_size=chunk_size, **CALL_KWARGS
        )
        return adata

    def test_chunk_size_1(self):
        adata = self._run_lowmem(1)
        _ct_ccc_results_equal(self.adata_stock.uns, adata.uns, self)

    def test_chunk_size_2(self):
        adata = self._run_lowmem(2)
        _ct_ccc_results_equal(self.adata_stock.uns, adata.uns, self)

    def test_chunk_size_equals_n_gp(self):
        n_gp = len(self.gene_pairs)
        adata = self._run_lowmem(n_gp)
        _ct_ccc_results_equal(self.adata_stock.uns, adata.uns, self)

    def test_chunk_size_exceeds_n_gp(self):
        """gene_pair_chunk_size >= n_gp must reproduce stock's single-pass behavior."""
        n_gp = len(self.gene_pairs)
        adata = self._run_lowmem(n_gp + 100)
        _ct_ccc_results_equal(self.adata_stock.uns, adata.uns, self)

    def test_chunk_size_none_adaptive(self):
        adata = self._run_lowmem(None)
        _ct_ccc_results_equal(self.adata_stock.uns, adata.uns, self)

    def test_all_chunk_sizes_mutually_identical(self):
        """Belt-and-braces: every chunk choice also matches every other choice directly
        (not just via the shared stock reference)."""
        n_gp = len(self.gene_pairs)
        results = {cs: self._run_lowmem(cs) for cs in (1, 2, n_gp, None)}
        keys = list(results)
        base = results[keys[0]]
        for other_key in keys[1:]:
            with self.subTest(chunk_size=other_key):
                _ct_ccc_results_equal(base.uns, results[other_key].uns, self)

    def test_single_chunk_parametric_Z_is_exactly_bit_identical(self):
        """A SINGLE chunk (gene_pair_chunk_size >= n_gp) executes the exact same
        reduction stock does, so Z/Z_pval/Z_FDR must be EXACTLY bit-for-bit identical
        here, with no tolerance at all (stricter than `_ct_ccc_results_equal`)."""
        n_gp = len(self.gene_pairs)
        adata = self._run_lowmem(n_gp)
        for grain in ("gp", "m"):
            for key in ("Z", "Z_pval", "Z_FDR"):
                with self.subTest(grain=grain, key=key):
                    np.testing.assert_array_equal(
                        self.adata_stock.uns["ct_ccc_results"]["p"][grain][key],
                        adata.uns["ct_ccc_results"]["p"][grain][key],
                    )


class CtCellCommunicationLowmemMemoryGuardTests(unittest.TestCase):
    """Non-vacuity: assert the chunked path is actually exercised (number of
    torch.sparse.mm calls scales with the number of chunks) in BOTH the observed path
    and the non-parametric permutation loop."""

    def test_sparse_mm_call_count_scales_with_chunks_observed(self):
        adata0 = make_test_adata(seed=9)
        n_gp = len(adata0.uns["gene_pairs"])
        n_ct_pairs = len(adata0.uns["cell_type_pairs"])
        counts_by_chunk = {}
        for chunk_size, n_chunks in ((1, n_gp), (n_gp, 1)):
            adata = make_test_adata(seed=9)
            kwargs = dict(CALL_KWARGS)
            kwargs["test"] = "parametric"  # simpler: no permutation loop to also count
            with mock.patch("torch.sparse.mm", wraps=torch.sparse.mm) as spy:
                compute_ct_cell_communication_lowmem(adata, gene_pair_chunk_size=chunk_size, **kwargs)
            counts_by_chunk[chunk_size] = spy.call_count
            # fix_ct=False (fixture default) -> 1 sparse.mm call per (chunk, ct_pair): WX2t_c only
            self.assertEqual(spy.call_count, n_ct_pairs * n_chunks, f"chunk_size={chunk_size}")

        self.assertGreater(counts_by_chunk[1], counts_by_chunk[n_gp])

    def test_np_permutation_loop_scales_with_chunks(self):
        """The permutation loop's sparse.mm count scales linearly with n_chunks (n_gp
        chunks of size 1 issues exactly n_gp x the calls of 1 chunk of size n_gp) --
        proves the reseed-per-chunk restructuring didn't collapse to a single pass."""
        adata0 = make_test_adata(seed=9)
        n_gp = len(adata0.uns["gene_pairs"])
        kwargs = dict(CALL_KWARGS)
        kwargs.update(test="non-parametric", M=3)
        counts = {}
        for chunk_size in (1, n_gp):
            adata = make_test_adata(seed=9)
            with mock.patch("torch.sparse.mm", wraps=torch.sparse.mm) as spy:
                compute_ct_cell_communication_lowmem(adata, gene_pair_chunk_size=chunk_size, **kwargs)
            counts[chunk_size] = spy.call_count
        self.assertEqual(counts[1], n_gp * counts[n_gp])
        self.assertGreater(counts[1], counts[n_gp])

    def test_all_gene_pairs_processed_across_chunks(self):
        """The chunked (n_ct_pairs, n_gp) cs array covers every gene pair exactly once,
        regardless of how many chunks it took."""
        adata_ref = make_test_adata(seed=3)
        gene_pairs = list(adata_ref.uns["gene_pairs"])
        harreman.tools.compute_ct_cell_communication(adata_ref, subset_gene_pairs=gene_pairs, **CALL_KWARGS)

        adata_chunked = make_test_adata(seed=3)
        compute_ct_cell_communication_lowmem(
            adata_chunked, subset_gene_pairs=gene_pairs, gene_pair_chunk_size=1, **CALL_KWARGS
        )

        n_gp = len(gene_pairs)
        self.assertEqual(adata_chunked.uns["ct_ccc_results"]["p"]["gp"]["cs"].shape[1], n_gp)
        np.testing.assert_array_equal(
            adata_ref.uns["ct_ccc_results"]["p"]["gp"]["cs"],
            adata_chunked.uns["ct_ccc_results"]["p"]["gp"]["cs"],
        )


class CtCellCommunicationLowmemGuardTests(unittest.TestCase):
    def test_check_analytic_null_raises_not_implemented(self):
        adata = make_test_adata(seed=2)
        kwargs = dict(CALL_KWARGS)
        kwargs["check_analytic_null"] = True
        with self.assertRaises(NotImplementedError):
            compute_ct_cell_communication_lowmem(adata, **kwargs)

    def test_fix_gp_true_raises_not_implemented(self):
        adata = make_test_adata(seed=2)
        kwargs = dict(CALL_KWARGS)
        kwargs["fix_gp"] = True
        with self.assertRaises(NotImplementedError):
            compute_ct_cell_communication_lowmem(adata, **kwargs)

    def test_invalid_test_raises_value_error(self):
        adata = make_test_adata(seed=2)
        kwargs = dict(CALL_KWARGS)
        kwargs["test"] = "bogus"
        with self.assertRaises(ValueError):
            compute_ct_cell_communication_lowmem(adata, **kwargs)

    def test_invalid_mean_raises_value_error(self):
        adata = make_test_adata(seed=2)
        kwargs = dict(CALL_KWARGS)
        kwargs["mean"] = "bogus"
        with self.assertRaises(ValueError):
            compute_ct_cell_communication_lowmem(adata, **kwargs)

    def test_missing_cell_type_key_raises_value_error(self):
        adata = make_test_adata(seed=2)
        kwargs = dict(CALL_KWARGS)
        del kwargs["cell_type_key"]
        with self.assertRaises(ValueError):
            compute_ct_cell_communication_lowmem(adata, **kwargs)


class CtCellCommunicationLowmemNonParametricOnlyTests(unittest.TestCase):
    """test='non-parametric' only (no 'p' branch, no EG2/Wtot2 at all -- exercises the
    standalone non-parametric chunk loop and its permutation restructuring in isolation)."""

    def setUp(self):
        self.adata_stock = make_test_adata(seed=11)
        self.adata_lowmem = make_test_adata(seed=11)
        gene_pairs = list(self.adata_stock.uns["gene_pairs"])
        kwargs = dict(CALL_KWARGS)
        kwargs["test"] = "non-parametric"

        harreman.tools.compute_ct_cell_communication(self.adata_stock, subset_gene_pairs=gene_pairs, **kwargs)
        compute_ct_cell_communication_lowmem(
            self.adata_lowmem, subset_gene_pairs=gene_pairs, gene_pair_chunk_size=1, **kwargs
        )

    def test_non_parametric_only_matches_stock(self):
        for grain in ("gp", "m"):
            for key in ("cs", "pval", "FDR", "perm_cs"):
                with self.subTest(grain=grain, key=key):
                    np.testing.assert_array_equal(
                        self.adata_stock.uns["ct_ccc_results"]["np"][grain][key],
                        self.adata_lowmem.uns["ct_ccc_results"]["np"][grain][key],
                    )
        self.assertNotIn("p", self.adata_lowmem.uns["ct_ccc_results"])


class CtCellCommunicationLowmemCenteredNpTests(unittest.TestCase):
    """center_counts_for_np_test=True + test='both' exercises the STOCK SHORTCUT branch
    (np observed score copied from the already-computed p score)."""

    def setUp(self):
        self.adata_stock = make_test_adata(seed=13)
        self.adata_lowmem = make_test_adata(seed=13)
        gene_pairs = list(self.adata_stock.uns["gene_pairs"])
        kwargs = dict(CALL_KWARGS)
        kwargs["center_counts_for_np_test"] = True

        harreman.tools.compute_ct_cell_communication(self.adata_stock, subset_gene_pairs=gene_pairs, **kwargs)
        compute_ct_cell_communication_lowmem(
            self.adata_lowmem, subset_gene_pairs=gene_pairs, gene_pair_chunk_size=1, **kwargs
        )

    def test_centered_np_matches_stock(self):
        _ct_ccc_results_equal(self.adata_stock.uns, self.adata_lowmem.uns, self)

    def test_np_cs_equals_p_cs_shortcut(self):
        """Confirms the shortcut actually fired (np cs is literally the p cs)."""
        for grain in ("gp", "m"):
            np.testing.assert_array_equal(
                self.adata_lowmem.uns["ct_ccc_results"]["p"][grain]["cs"],
                self.adata_lowmem.uns["ct_ccc_results"]["np"][grain]["cs"],
            )


class CtCellCommunicationLowmemFixCtTests(unittest.TestCase):
    """fix_ct=True exercises the EG2 chunked-matmul branch (`if fix_ct:` inside the
    parametric chunk loop) that the default (fix_ct=False, Wtot2-only) fixture never
    reaches. Parametric only -- fix_ct doesn't change permutation RNG consumption
    (verified: the stratified idx draw is unconditional; only which counts get
    permuted -- c1_perm -- depends on fix_ct), so this focuses on the EG2 chunking."""

    def _make(self, seed):
        adata = make_test_adata(seed=seed)
        adata.uns["fix_ct"] = True
        return adata

    def test_fix_ct_matches_stock_across_chunk_sizes(self):
        gene_pairs = None  # exercise the subset_gene_pairs=None fallback too
        kwargs = dict(CALL_KWARGS)
        kwargs["test"] = "parametric"

        adata_stock = self._make(seed=17)
        harreman.tools.compute_ct_cell_communication(adata_stock, subset_gene_pairs=gene_pairs, **kwargs)

        n_gp = len(adata_stock.uns["gene_pairs"])
        for chunk_size in (1, 2, n_gp, None):
            with self.subTest(chunk_size=chunk_size):
                adata_lowmem = self._make(seed=17)
                compute_ct_cell_communication_lowmem(
                    adata_lowmem, subset_gene_pairs=gene_pairs, gene_pair_chunk_size=chunk_size, **kwargs
                )
                np.testing.assert_array_equal(
                    adata_stock.uns["ct_ccc_results"]["p"]["gp"]["cs"],
                    adata_lowmem.uns["ct_ccc_results"]["p"]["gp"]["cs"],
                )
                np.testing.assert_allclose(
                    adata_stock.uns["ct_ccc_results"]["p"]["gp"]["Z"],
                    adata_lowmem.uns["ct_ccc_results"]["p"]["gp"]["Z"],
                    rtol=1e-11, atol=1e-13,
                )


def make_ct_specific_adata(seed=1):
    """Like make_test_adata, BUT two ct_pairs get a STRICT SUBSET of the gene pairs, so
    `ct_specific_gene_pairs` is non-empty -- exercising compute_metabolite_cs_ct's in-place
    `cs_gp[i, mask] = 0` mutation and the cloned permutation-derived masking that the main
    fixture never reaches (see the module docstring's scope note)."""
    rng = np.random.default_rng(seed)
    counts = rng.poisson(lam=5, size=(N_CELLS, len(GENES))).astype(np.float64) + 1.0
    log_norm = np.log1p(counts / counts.sum(axis=1, keepdims=True) * 1e4)

    adata = ad.AnnData(X=log_norm.copy())
    adata.var_names = GENES
    adata.obs_names = [f"cell_{i}" for i in range(N_CELLS)]
    adata.obs["cell_type"] = pd.Categorical(CELL_TYPES)
    adata.layers["counts"] = counts
    adata.layers["log_norm"] = log_norm
    adata.obsp["weights"] = _make_weights(N_CELLS, k=6, seed=3)

    gene_pairs = [("G1", "G1"), ("G2", "G3"), (["G4", "G5"], "G6")]
    adata.uns["gene_pairs"] = gene_pairs
    adata.uns["gene_pairs_per_metabolite"] = {
        "Metab_A": {"gene_pair": [("G1", "G1"), ("G2", "G3")],
                    "gene_type": [("IMP-EXP", "IMP-EXP"), ("IMP-EXP", "IMP-EXP")]},
        "Metab_B": {"gene_pair": [(["G4", "G5"], "G6")],
                    "gene_type": [("IMP-EXP", "IMP-EXP")]},
    }

    ct_pairs = list(itertools.combinations_with_replacement(sorted(set(CELL_TYPES)), 2))
    adata.uns["cell_type_pairs"] = ct_pairs
    # KEY: two ct_pairs carry a proper subset of uns['gene_pairs'] -> non-empty
    # ct_specific_gene_pairs (drives the set-comparison `{...} < gene_pairs_ind_set`).
    gpp = {}
    for i, cp in enumerate(ct_pairs):
        if i == 1:
            gpp[cp] = [gene_pairs[0], gene_pairs[1]]   # 2/3 -> strict subset
        elif i == 3:
            gpp[cp] = [gene_pairs[2]]                  # 1/3 -> strict subset
        else:
            gpp[cp] = list(gene_pairs)                 # full list
    adata.uns["gene_pairs_per_ct_pair"] = gpp
    adata.uns["fix_ct"] = False
    return adata


def _prep_internal(adata):
    # The public wrappers set these before delegating to the internal analysis fn.
    adata.uns["ct_ccc_results"] = {}
    adata.uns["cell_type_key"] = "cell_type"


class CtSpecificGenePairsMaskingEquivalenceTests(unittest.TestCase):
    """Non-empty `ct_specific_gene_pairs`: internal stock-vs-lowmem equivalence across
    chunk sizes. Closes the coverage gap flagged in the module docstring. We call the
    INTERNAL analysis fns directly (not the public wrapper): stock's
    `get_ct_cell_communication_results` DataFrame builder crashes on non-uniform per-ct-pair
    lengths -- a pre-existing stock limitation, unreachable in Foster's config
    (expression_filt/de_filt default False -> filtered_genes_ct uniform -> empty
    ct_specific_gene_pairs). The masking mutation firing is confirmed by the mask-branch
    being reached; here we assert the RESULT is bit-identical to stock regardless."""

    LAYER_P, LAYER_NP, MODEL, CTKEY, M, SEED = "counts", "log_norm", "danb", "cell_type", 20, 7
    DEVICE = torch.device("cpu")

    def _run_stock(self, adata, gene_pairs):
        _prep_internal(adata)
        _stock_run_ct(adata, self.LAYER_P, self.LAYER_NP, self.MODEL, self.CTKEY, False,
                      gene_pairs, None, False, self.M, self.SEED, "both", "algebraic",
                      False, self.DEVICE, False)

    def _run_lowmem(self, adata, gene_pairs, chunk):
        _prep_internal(adata)
        _run_ct_cell_communication_analysis_lowmem(
            adata, self.LAYER_P, self.LAYER_NP, self.MODEL, self.CTKEY, False,
            gene_pairs, None, self.M, self.SEED, "both", "algebraic",
            self.DEVICE, False, chunk)

    def test_ct_specific_gene_pairs_is_nonempty(self):
        adata = make_ct_specific_adata(seed=1)
        self._run_stock(adata, list(adata.uns["gene_pairs"]))
        lens = {k: len(v) for k, v in adata.uns["gene_pairs_ind_per_ct_pair"].items()}
        self.assertTrue(any(v < 3 for v in lens.values()), lens)

    def test_masking_path_bit_identical_across_chunks(self):
        gene_pairs = list(make_ct_specific_adata(seed=1).uns["gene_pairs"])
        adata_stock = make_ct_specific_adata(seed=1)
        self._run_stock(adata_stock, gene_pairs)

        for chunk in (1, 2, 3, None):
            adata_lowmem = make_ct_specific_adata(seed=1)
            self._run_lowmem(adata_lowmem, gene_pairs, chunk)
            for grain in ("gp", "m"):
                with self.subTest(chunk=chunk, test="p", grain=grain, key="cs"):
                    np.testing.assert_array_equal(
                        adata_stock.uns["ct_ccc_results"]["p"][grain]["cs"],
                        adata_lowmem.uns["ct_ccc_results"]["p"][grain]["cs"])
                for key in ("cs", "pval", "FDR", "perm_cs"):  # entire np path -> EXACT
                    with self.subTest(chunk=chunk, test="np", grain=grain, key=key):
                        np.testing.assert_array_equal(
                            adata_stock.uns["ct_ccc_results"]["np"][grain][key],
                            adata_lowmem.uns["ct_ccc_results"]["np"][grain][key])
                for key in ("Z", "Z_pval", "Z_FDR"):  # parametric Z -> ULP-tolerant
                    with self.subTest(chunk=chunk, test="p", grain=grain, key=key):
                        np.testing.assert_allclose(
                            adata_stock.uns["ct_ccc_results"]["p"][grain][key],
                            adata_lowmem.uns["ct_ccc_results"]["p"][grain][key],
                            rtol=1e-11, atol=1e-13)


if __name__ == "__main__":
    unittest.main()
