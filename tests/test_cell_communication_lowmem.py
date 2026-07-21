"""Tier-0/1 equivalence test for `metab_processing/Harreman/cell_communication_lowmem.py`
(CU-A). No GPU/CUDA/Savio/real-harreman needed: `harreman` is a small local fake package
(`tests/fixtures/fake_harreman/`) vendoring the *real* stock
`compute_interacting_cell_scores` verbatim from `DataForClaude/cell_communication.py`,
PLUS the real (not stubbed) `counts_from_anndata`, `make_weights_non_redundant`, and
`standardize_counts`/DANB helpers vendored from a local harreman clone (see
`tests/fixtures/fake_harreman/harreman/tools.py`'s module docstring for exact provenance).
So this asserts the low-mem drop-in reproduces the stock *algorithm* bit-for-bit (observed
scores) / near-bit-for-bit (permutation p-values -- see the float32/float64 note below)
using real numerics, not just "some numbers came out."

The fixture uses `uns['model'] = 'danb'` (matching what `harreman_funcs.py` actually uses)
so the parametric branch (`CellCommunicationLowmemParametricDanbTests` below) exercises
real DANB standardization. The main equivalence test above it uses
`test='non-parametric'` only (matching how `nbhd_scores.py` actually calls the drop-in),
which does NOT invoke `standardize_counts` at all (`center_counts_for_np_test=False`) --
that's why the DANB path needs its own dedicated test.

`conftest.py` puts `metab_processing/Harreman` and the fake `harreman` package on
`sys.path`, so plain `import harreman` / `import cell_communication_lowmem` both resolve.
"""
import unittest

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

import harreman
from cell_communication_lowmem import compute_interacting_cell_scores_lowmem

N_CELLS = 50
GENES = [f"G{i}" for i in range(1, 9)]  # G1..G8 (G7, G8 unused filler)


def _make_weights(n_cells, k=4, seed=0):
    """Deterministic small-degree sparse adjacency (stand-in for a spatial KNN graph;
    the real weight math doesn't matter here, see fake_harreman's
    `make_weights_non_redundant` docstring -- both code paths under test consume this
    matrix identically)."""
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
    """Tiny synthetic AnnData exercising three gene-pair shapes across two metabolites:
      - a homotypic (same-gene) pair:      ("G1", "G1")             -> Metab_A
      - a plain heterotypic pair:          ("G2", "G3")              -> Metab_A
      - a heterodimer tuple pair:          (("G4", "G5"), "G6")      -> Metab_B
    Mirrors the uns/obsp contract `compute_interacting_cell_scores` reads (see
    DataForClaude/cell_communication.py lines 1473-1905): `uns['model']`, `uns['mean']`,
    `uns['gene_pairs']`, `uns['gene_pairs_per_metabolite']`, `uns['genes']`,
    `uns['ccc_results']['cell_com_df_{gp,m}_sig']`, `uns['layer_key_np_test']`,
    `obsp['weights']`, and a matching layer.
    """
    rng = np.random.default_rng(seed)
    n_genes = len(GENES)
    log_norm = rng.gamma(shape=2.0, scale=1.0, size=(n_cells, n_genes))

    adata = ad.AnnData(X=log_norm.copy())
    adata.var_names = GENES
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.layers["log_norm"] = log_norm

    adata.obsp["weights"] = _make_weights(n_cells, k=4, seed=seed)

    adata.uns["model"] = "danb"  # matches what `harreman_funcs.py` actually uses
    adata.uns["mean"] = "algebraic"
    adata.uns["layer_key_np_test"] = "log_norm"
    adata.uns["layer_key_p_test"] = None
    adata.uns["genes"] = list(GENES)

    adata.uns["gene_pairs"] = [
        ["G1", "G1"],
        ["G2", "G3"],
        [["G4", "G5"], "G6"],
    ]
    adata.uns["gene_pairs_per_metabolite"] = {
        "Metab_A": {
            "gene_pair": [["G1", "G1"], ["G2", "G3"]],
            "gene_type": [["IMP-EXP", "IMP-EXP"], ["IMP-EXP", "IMP-EXP"]],
        },
        "Metab_B": {
            "gene_pair": [[["G4", "G5"], "G6"]],
            "gene_type": [["IMP-EXP", "IMP-EXP"]],
        },
    }
    adata.uns["ccc_results"] = {
        "cell_com_df_gp_sig": pd.DataFrame(
            {"Gene 1": ["G1", "G2", ["G4", "G5"]], "Gene 2": ["G1", "G3", "G6"]}
        ),
        "cell_com_df_m_sig": pd.DataFrame({"Metabolite": ["Metab_A", "Metab_B"]}),
    }
    return adata


# Matches how `nbhd_scores.compute_nbhd_scores` actually calls the drop-in.
CALL_KWARGS = dict(
    center_counts_for_np_test=False,
    test="non-parametric",
    restrict_significance="both",
    compute_significance="non-parametric",
    M=50,
    seed=7,
    check_analytic_null=False,
    verbose=False,
)


class CellCommunicationLowmemEquivalenceTests(unittest.TestCase):
    def setUp(self):
        self.adata_stock = make_test_adata(seed=1)
        self.adata_lowmem = make_test_adata(seed=1)

        harreman.tools.compute_interacting_cell_scores(self.adata_stock, **CALL_KWARGS)
        compute_interacting_cell_scores_lowmem(self.adata_lowmem, **CALL_KWARGS)

        self.stock_np = self.adata_stock.uns["interacting_cell_results"]["np"]
        self.lowmem_np = self.adata_lowmem.uns["interacting_cell_results"]["np"]

    def test_gene_pair_bookkeeping_matches(self):
        """Sanity check the shared filtering/indexing logic (not the memory fix)
        produced identical bookkeeping before comparing scores."""
        self.assertEqual(self.adata_stock.uns["gene_pairs_sig"], self.adata_lowmem.uns["gene_pairs_sig"])
        self.assertEqual(self.adata_stock.uns["gene_pairs_sig_ind"], self.adata_lowmem.uns["gene_pairs_sig_ind"])
        self.assertEqual(self.adata_stock.uns["metabolites"], self.adata_lowmem.uns["metabolites"])
        self.assertEqual(self.adata_stock.uns["gene_pairs_sig_names"], self.adata_lowmem.uns["gene_pairs_sig_names"])
        # all three designed gene-pair shapes actually survived the significance filter
        self.assertEqual(len(self.adata_lowmem.uns["gene_pairs_sig"]), 3)

    def test_observed_scores_are_bit_identical(self):
        """`cs` (the observed, non-permuted score) involves no RNG and no float32 cast
        in either path -> exact equality."""
        for grain in ("gp", "m"):
            with self.subTest(grain=grain):
                np.testing.assert_array_equal(
                    self.stock_np[grain]["cs"], self.lowmem_np[grain]["cs"]
                )

    def test_pval_and_fdr_are_bit_identical(self):
        """The drop-in replicates stock's float32 cast of the exceedance count
        (`(x + 1).float() / (M + 1)`, cell_communication_lowmem.py); same seed -> same
        permutation sequence -> same integer counts -> pval/FDR/cs_sig_* are bit-for-bit
        equal to stock (not merely close). Exact equality is the strongest guard against
        any sign/arm-swap/off-by-one drift in the incremental accumulation."""
        for grain in ("gp", "m"):
            with self.subTest(grain=grain):
                for key in ("pval", "FDR", "cs_sig_pval", "cs_sig_FDR"):
                    np.testing.assert_array_equal(
                        self.stock_np[grain][key], self.lowmem_np[grain][key],
                        err_msg=f"mismatch in ['np']['{grain}']['{key}']",
                    )

    def test_lowmem_does_not_store_raw_permutation_arrays(self):
        """The whole point of the drop-in: never materialize (cells, pairs, M)."""
        for grain in ("gp", "m"):
            self.assertNotIn("perm_cs_a", self.lowmem_np[grain])
            self.assertNotIn("perm_cs_b", self.lowmem_np[grain])
        # sanity: confirm the *stock* function (which we did NOT patch) still does --
        # otherwise this test would be vacuous.
        for grain in ("gp", "m"):
            self.assertIn("perm_cs_a", self.stock_np[grain])
            self.assertIn("perm_cs_b", self.stock_np[grain])


class CellCommunicationLowmemBothTests(unittest.TestCase):
    """`test='both'` is exactly what `run_harreman.ipynb` calls in production. This runs
    the parametric AND non-parametric branches in a single pass (exercising the p->np
    ordering) and asserts every stored array matches stock. With the float32 cast now
    replicated, even the permutation p-values are bit-for-bit equal."""

    def setUp(self):
        self.adata_stock = make_test_adata(seed=5)
        self.adata_lowmem = make_test_adata(seed=5)
        # `layer_key_p_test=None` -> parametric path reads adata.X (see make_test_adata)
        kwargs = dict(CALL_KWARGS)
        kwargs.update(test="both", compute_significance="both")

        harreman.tools.compute_interacting_cell_scores(self.adata_stock, **kwargs)
        compute_interacting_cell_scores_lowmem(self.adata_lowmem, **kwargs)

        self.stock = self.adata_stock.uns["interacting_cell_results"]
        self.lowmem = self.adata_lowmem.uns["interacting_cell_results"]

    def test_both_branches_match_stock_exactly(self):
        checks = {
            "p": ("cs", "Z", "Z_pval", "Z_FDR", "cs_sig_pval", "cs_sig_FDR"),
            "np": ("cs", "pval", "FDR", "cs_sig_pval", "cs_sig_FDR"),
        }
        for test_key, keys in checks.items():
            for grain in ("gp", "m"):
                for key in keys:
                    with self.subTest(test=test_key, grain=grain, key=key):
                        np.testing.assert_array_equal(
                            self.stock[test_key][grain][key], self.lowmem[test_key][grain][key],
                            err_msg=f"mismatch in ['{test_key}']['{grain}']['{key}']",
                        )

    def test_lowmem_still_drops_perm_arrays_in_both_mode(self):
        for grain in ("gp", "m"):
            self.assertNotIn("perm_cs_a", self.lowmem["np"][grain])
            self.assertNotIn("perm_cs_b", self.lowmem["np"][grain])


class CellCommunicationLowmemGuardTests(unittest.TestCase):
    def test_check_analytic_null_raises_not_implemented(self):
        adata = make_test_adata(seed=2)
        kwargs = dict(CALL_KWARGS)
        kwargs["check_analytic_null"] = True
        with self.assertRaises(NotImplementedError):
            compute_interacting_cell_scores_lowmem(adata, **kwargs)


class CellCommunicationLowmemParametricDanbTests(unittest.TestCase):
    """The parametric ('p') branch is reproduced verbatim in the low-mem drop-in (no
    permutation / no M axis, so no memory fix needed there) -- but it's the ONLY branch
    that actually calls `standardize_counts` when `center_counts_for_np_test=False`
    (see `CALL_KWARGS` above), so the main non-parametric-only equivalence test never
    exercises real DANB standardization. This test does, using `test='parametric'`."""

    def setUp(self):
        self.adata_stock = make_test_adata(seed=3)
        self.adata_lowmem = make_test_adata(seed=3)

        kwargs = dict(CALL_KWARGS)
        kwargs.update(test="parametric", compute_significance="parametric")

        harreman.tools.compute_interacting_cell_scores(self.adata_stock, **kwargs)
        compute_interacting_cell_scores_lowmem(self.adata_lowmem, **kwargs)

        self.stock_p = self.adata_stock.uns["interacting_cell_results"]["p"]
        self.lowmem_p = self.adata_lowmem.uns["interacting_cell_results"]["p"]

    def test_danb_standardized_parametric_scores_match_exactly(self):
        """No RNG and no float32 cast anywhere in the parametric path (unlike the
        permutation test), so stock and low-mem should match bit-for-bit here."""
        for grain in ("gp", "m"):
            with self.subTest(grain=grain):
                for key in ("cs", "Z", "Z_pval", "Z_FDR", "cs_sig_pval", "cs_sig_FDR"):
                    np.testing.assert_array_equal(
                        self.stock_p[grain][key], self.lowmem_p[grain][key],
                        err_msg=f"mismatch in interacting_cell_results['p']['{grain}']['{key}']",
                    )

    def test_danb_produced_finite_scores(self):
        """Sanity check the real DANB math didn't degenerate to NaN/Inf on this tiny
        synthetic fixture (would silently make the equality assertions above vacuous --
        NaN == NaN is False, so a NaN-vs-NaN 'match' would actually fail loudly, but a
        NaN-vs-non-NaN mismatch could be masked by unrelated float noise elsewhere)."""
        for grain in ("gp", "m"):
            self.assertTrue(np.all(np.isfinite(self.stock_p[grain]["cs"])))
            self.assertTrue(np.all(np.isfinite(self.stock_p[grain]["Z"])))


if __name__ == "__main__":
    unittest.main()
