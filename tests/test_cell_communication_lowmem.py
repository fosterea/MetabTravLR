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


def make_shared_pair_test_adata(n_cells=N_CELLS, seed=0):
    """Second fixture for the CU-E (Option B chunking) equivalence sweep. Exercises the
    two properties the single-chunk fixture above (`make_test_adata`) cannot:
      (a) a metabolite whose gene pairs land in DIFFERENT gene-pair chunks (chunk size 1
          or 2 forces this for any metabolite owning >1 pair), and
      (b) a gene pair SHARED by >=2 metabolites (many-to-many) -- pair index 1 (G2, G3)
          below belongs to both Metab_A and Metab_C.

    Four gene pairs / three metabolites:
      - pair 0 ("G1","G1")            homotypic          -> Metab_A only
      - pair 1 ("G2","G3")            heterotypic         -> Metab_A AND Metab_C (shared)
      - pair 2 (("G4","G5"),"G6")     heterodimer         -> Metab_B only
      - pair 3 ("G3","G2")            heterotypic (rev.)  -> Metab_C only
    Metab_A = {0, 1}, Metab_B = {2}, Metab_C = {1, 3} -- so with gene_pair_chunk_size=1,
    Metab_A's and Metab_C's pairs each span two different chunks, and pair 1 is shared.
    """
    rng = np.random.default_rng(seed)
    n_genes = len(GENES)
    log_norm = rng.gamma(shape=2.0, scale=1.0, size=(n_cells, n_genes))

    adata = ad.AnnData(X=log_norm.copy())
    adata.var_names = GENES
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.layers["log_norm"] = log_norm

    adata.obsp["weights"] = _make_weights(n_cells, k=4, seed=seed)

    adata.uns["model"] = "danb"
    adata.uns["mean"] = "algebraic"
    adata.uns["layer_key_np_test"] = "log_norm"
    adata.uns["layer_key_p_test"] = None
    adata.uns["genes"] = list(GENES)

    adata.uns["gene_pairs"] = [
        ["G1", "G1"],
        ["G2", "G3"],
        [["G4", "G5"], "G6"],
        ["G3", "G2"],
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
        "Metab_C": {
            "gene_pair": [["G2", "G3"], ["G3", "G2"]],
            "gene_type": [["IMP-EXP", "IMP-EXP"], ["IMP-EXP", "IMP-EXP"]],
        },
    }
    adata.uns["ccc_results"] = {
        "cell_com_df_gp_sig": pd.DataFrame(
            {
                "Gene 1": ["G1", "G2", ["G4", "G5"], "G3"],
                "Gene 2": ["G1", "G3", "G6", "G2"],
            }
        ),
        "cell_com_df_m_sig": pd.DataFrame({"Metabolite": ["Metab_A", "Metab_B", "Metab_C"]}),
    }
    return adata


def make_many_pairs_test_adata(n_cells=N_CELLS, seed=0):
    """Third fixture: like `make_shared_pair_test_adata` but gives Metab_C a THIRD gene
    pair (index 4, using the previously-unused G7/G8), so one metabolite owns >=3 gene
    pairs -- real metabolites go up to 91 (05_harreman_reference.md sec.4a). Used by the
    `center_counts_for_np_test=True` sweep so the standardized-union path and the
    'p'-shortcut copy are exercised on a metabolite wide enough to be representative.

    Five gene pairs / three metabolites:
      - pair 0 ("G1","G1")            homotypic          -> Metab_A only
      - pair 1 ("G2","G3")            heterotypic         -> Metab_A AND Metab_C (shared)
      - pair 2 (("G4","G5"),"G6")     heterodimer         -> Metab_B only
      - pair 3 ("G3","G2")            heterotypic (rev.)  -> Metab_C only
      - pair 4 ("G7","G8")            heterotypic         -> Metab_C only
    Metab_A = {0, 1}, Metab_B = {2}, Metab_C = {1, 3, 4}.
    """
    rng = np.random.default_rng(seed)
    n_genes = len(GENES)
    log_norm = rng.gamma(shape=2.0, scale=1.0, size=(n_cells, n_genes))

    adata = ad.AnnData(X=log_norm.copy())
    adata.var_names = GENES
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.layers["log_norm"] = log_norm

    adata.obsp["weights"] = _make_weights(n_cells, k=4, seed=seed)

    adata.uns["model"] = "danb"
    adata.uns["mean"] = "algebraic"
    adata.uns["layer_key_np_test"] = "log_norm"
    adata.uns["layer_key_p_test"] = None
    adata.uns["genes"] = list(GENES)

    adata.uns["gene_pairs"] = [
        ["G1", "G1"],
        ["G2", "G3"],
        [["G4", "G5"], "G6"],
        ["G3", "G2"],
        ["G7", "G8"],
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
        "Metab_C": {
            "gene_pair": [["G2", "G3"], ["G3", "G2"], ["G7", "G8"]],
            "gene_type": [["IMP-EXP", "IMP-EXP"], ["IMP-EXP", "IMP-EXP"], ["IMP-EXP", "IMP-EXP"]],
        },
    }
    adata.uns["ccc_results"] = {
        "cell_com_df_gp_sig": pd.DataFrame(
            {
                "Gene 1": ["G1", "G2", ["G4", "G5"], "G3", "G7"],
                "Gene 2": ["G1", "G3", "G6", "G2", "G8"],
            }
        ),
        "cell_com_df_m_sig": pd.DataFrame({"Metabolite": ["Metab_A", "Metab_B", "Metab_C"]}),
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


# ============================================================================================
# CU-E: Option B (gene-pair + metabolite chunking) equivalence sweep.
#
# The chunked `np` path is compared against TRUE stock (`harreman.tools.compute_
# interacting_cell_scores`), not just the unchunked lowmem function, so the whole chain
# (including the pre-CU-E lowmem code that stays untouched) is pinned to stock -- exactly
# what the CU-E task spec asks for. Two fixtures:
#   - `make_test_adata` (3 gene pairs / 2 metabolites; already used above).
#   - `make_shared_pair_test_adata` (4 gene pairs / 3 metabolites; adds a metabolite whose
#     pairs span different gene-pair chunks AND a gene pair shared by 2 metabolites).
# ============================================================================================

_CHUNK_FIXTURES = {
    "three_pairs_two_metabs": (make_test_adata, 3, 2),
    "shared_pair_many_to_many": (make_shared_pair_test_adata, 4, 3),
}


class CellCommunicationLowmemChunkedEquivalenceTests(unittest.TestCase):
    """Sweeps ``gene_pair_chunk_size`` x ``metabolite_chunk_size`` (incl. 1, 2, the exact
    full size, and ``None``/adaptive) and asserts every stored non-parametric array is
    bit-for-bit identical to stock in every configuration."""

    def _assert_np_matches_stock(self, adata_factory, gp_chunk, m_chunk, seed):
        adata_stock = adata_factory(seed=seed)
        adata_lowmem = adata_factory(seed=seed)

        harreman.tools.compute_interacting_cell_scores(adata_stock, **CALL_KWARGS)
        lowmem_kwargs = dict(CALL_KWARGS)
        lowmem_kwargs["gene_pair_chunk_size"] = gp_chunk
        lowmem_kwargs["metabolite_chunk_size"] = m_chunk
        compute_interacting_cell_scores_lowmem(adata_lowmem, **lowmem_kwargs)

        stock_np = adata_stock.uns["interacting_cell_results"]["np"]
        lowmem_np = adata_lowmem.uns["interacting_cell_results"]["np"]
        for grain in ("gp", "m"):
            for key in ("cs", "pval", "FDR", "cs_sig_pval", "cs_sig_FDR"):
                np.testing.assert_array_equal(
                    stock_np[grain][key], lowmem_np[grain][key],
                    err_msg=(
                        f"mismatch in ['np']['{grain}']['{key}'] "
                        f"(gene_pair_chunk_size={gp_chunk}, metabolite_chunk_size={m_chunk})"
                    ),
                )

    def test_chunk_sweep_non_parametric(self):
        """Primary sweep -- this is the only path `nbhd_scores.py` actually runs."""
        for fixture_name, (factory, n_gp, n_m) in _CHUNK_FIXTURES.items():
            for gp_chunk in (1, 2, n_gp, None):
                for m_chunk in (1, 2, n_m, None):
                    with self.subTest(fixture=fixture_name, gene_pair_chunk_size=gp_chunk, metabolite_chunk_size=m_chunk):
                        self._assert_np_matches_stock(factory, gp_chunk, m_chunk, seed=11)

    def test_chunk_sweep_both_smoke(self):
        """`test='both'` smoke at a few chunk sizes: exercises the parametric branch
        (untouched by CU-E) interleaved with the chunked non-parametric branch, incl. the
        `center_counts_for_np_test`-independent p->np ordering."""
        for gp_chunk, m_chunk in [(1, 1), (2, None), (None, 2), (4, 3)]:
            with self.subTest(gene_pair_chunk_size=gp_chunk, metabolite_chunk_size=m_chunk):
                adata_stock = make_shared_pair_test_adata(seed=13)
                adata_lowmem = make_shared_pair_test_adata(seed=13)
                kwargs = dict(CALL_KWARGS)
                kwargs.update(test="both", compute_significance="both")
                harreman.tools.compute_interacting_cell_scores(adata_stock, **kwargs)
                lowmem_kwargs = dict(kwargs)
                lowmem_kwargs["gene_pair_chunk_size"] = gp_chunk
                lowmem_kwargs["metabolite_chunk_size"] = m_chunk
                compute_interacting_cell_scores_lowmem(adata_lowmem, **lowmem_kwargs)

                checks = {
                    "p": ("cs", "Z", "Z_pval", "Z_FDR", "cs_sig_pval", "cs_sig_FDR"),
                    "np": ("cs", "pval", "FDR", "cs_sig_pval", "cs_sig_FDR"),
                }
                for test_key, keys in checks.items():
                    for grain in ("gp", "m"):
                        for key in keys:
                            np.testing.assert_array_equal(
                                adata_stock.uns["interacting_cell_results"][test_key][grain][key],
                                adata_lowmem.uns["interacting_cell_results"][test_key][grain][key],
                                err_msg=f"mismatch in ['{test_key}']['{grain}']['{key}'] (gp_chunk={gp_chunk}, m_chunk={m_chunk})",
                            )

    def test_chunk_sweep_centered_shortcut(self):
        """Dedicated sweep for `center_counts_for_np_test=True, test='both'` -- previously
        UNEXERCISED by any fixture/kwarg combo in this file, so neither the chunked
        `standardize_counts` calls (Pass 1's `counts_1c/counts_2c`, Pass 2's
        `counts_1u/counts_2u`) nor the 'p'-shortcut COPY (`np['gp']['cs']`/`np['m']['cs']`
        == a copy of the parametric 'p' cs, not a recompute) were ever tested. This is
        exactly the combination in which review caught a real bug: the `m`-grain shortcut
        copy must be sliced (not recomputed) for the Pass-2 exceedance threshold too. Uses
        `make_many_pairs_test_adata` (Metab_C owns 3 gene pairs) so the copied array is
        sliced across a real multi-pair metabolite chunk boundary."""
        factory, n_gp, n_m = make_many_pairs_test_adata, 5, 3
        for gp_chunk in (1, 2, n_gp, None):
            for m_chunk in (1, 2, n_m, None):
                with self.subTest(gene_pair_chunk_size=gp_chunk, metabolite_chunk_size=m_chunk):
                    adata_stock = factory(seed=19)
                    adata_lowmem = factory(seed=19)
                    kwargs = dict(CALL_KWARGS)
                    kwargs.update(
                        test="both",
                        compute_significance="both",
                        center_counts_for_np_test=True,
                    )
                    harreman.tools.compute_interacting_cell_scores(adata_stock, **kwargs)
                    lowmem_kwargs = dict(kwargs)
                    lowmem_kwargs["gene_pair_chunk_size"] = gp_chunk
                    lowmem_kwargs["metabolite_chunk_size"] = m_chunk
                    compute_interacting_cell_scores_lowmem(adata_lowmem, **lowmem_kwargs)

                    checks = {
                        "p": ("cs", "Z", "Z_pval", "Z_FDR", "cs_sig_pval", "cs_sig_FDR"),
                        "np": ("cs", "pval", "FDR", "cs_sig_pval", "cs_sig_FDR"),
                    }
                    for test_key, keys in checks.items():
                        for grain in ("gp", "m"):
                            for key in keys:
                                np.testing.assert_array_equal(
                                    adata_stock.uns["interacting_cell_results"][test_key][grain][key],
                                    adata_lowmem.uns["interacting_cell_results"][test_key][grain][key],
                                    err_msg=(
                                        f"mismatch in ['{test_key}']['{grain}']['{key}'] "
                                        f"(gp_chunk={gp_chunk}, m_chunk={m_chunk}, centered=True)"
                                    ),
                                )

    def test_many_to_many_pair_is_counted_for_both_metabolites(self):
        """Sanity check the fixture actually exercises what it claims: gene pair 1
        (G2, G3) contributes to BOTH Metab_A and Metab_C's observed cs_m -- i.e. the
        many-to-many bookkeeping (not just the numeric equality above) is real."""
        adata = make_shared_pair_test_adata(seed=17)
        compute_interacting_cell_scores_lowmem(
            adata, **{**CALL_KWARGS, "gene_pair_chunk_size": 1, "metabolite_chunk_size": 1}
        )
        metabolites = adata.uns["metabolites"]
        self.assertIn("Metab_A", metabolites)
        self.assertIn("Metab_C", metabolites)
        cs_gp = adata.uns["interacting_cell_results"]["np"]["gp"]["cs"]
        cs_m = adata.uns["interacting_cell_results"]["np"]["m"]["cs"]
        idx_a = metabolites.index("Metab_A")
        idx_c = metabolites.index("Metab_C")
        gene_pairs_sig = adata.uns["gene_pairs_sig"]
        shared_pair = ("G2", "G3")
        self.assertIn(shared_pair, gene_pairs_sig)
        shared_idx = gene_pairs_sig.index(shared_pair)
        # Metab_A = pairs {("G1","G1"), ("G2","G3")}; Metab_C = pairs {("G2","G3"), ("G3","G2")}
        pair_ga = gene_pairs_sig.index(("G1", "G1"))
        pair_gc = gene_pairs_sig.index(("G3", "G2"))
        np.testing.assert_allclose(cs_m[:, idx_a], cs_gp[:, pair_ga] + cs_gp[:, shared_idx])
        np.testing.assert_allclose(cs_m[:, idx_c], cs_gp[:, shared_idx] + cs_gp[:, pair_gc])


class CellCommunicationLowmemMemoryShapeGuardTests(unittest.TestCase):
    """Guards the actual memory claim (07_nbhd_percell_chunking_plan.md sec.7): with tiny
    chunk sizes, no gene-pair/metabolite-axis tensor wider than the chunk (or a single
    metabolite's own gene-pair union, for Pass 2) should ever pass through
    ``torch.sparse.mm`` -- the ACTUAL site where the pre-CU-E `(n_cells, n_gp)`
    intermediates lived (`WX2t`/`WtX2t` and their permutation-arm twins), not just the
    `torch.zeros`-allocated accumulators. Spying only on `zeros`/`zeros_like` (the
    original version of this guard) would sail right past a reintroduced full-width
    `sparse.mm` call, so this version spies `sparse.mm`'s dense operand/output widths too.

    Includes a POSITIVE CONTROL: the identical measurement re-run with large
    ("unchunked") chunk sizes must reach the full `n_gp` width -- proving the metric is
    actually capable of detecting an unbounded allocation, i.e. the bounded result for the
    chunked run is not just a vacuously-small measurement that would pass regardless of
    whether chunking worked."""

    def _measure_max_gp_axis_width(self, adata, gp_chunk, m_chunk):
        import torch
        import cell_communication_lowmem as cclm

        seen_widths = []

        def _record(t):
            if hasattr(t, "dim") and t.dim() == 2:
                seen_widths.append(t.shape[1])

        orig_zeros = torch.zeros
        orig_zeros_like = torch.zeros_like
        orig_sparse_mm = torch.sparse.mm

        def spy_zeros(*args, **kwargs):
            t = orig_zeros(*args, **kwargs)
            _record(t)
            return t

        def spy_zeros_like(*args, **kwargs):
            t = orig_zeros_like(*args, **kwargs)
            _record(t)
            return t

        def spy_sparse_mm(a, b, *args, **kwargs):
            # `a` is always the (n_cells, n_cells) sparse `weights` (or its transpose) in
            # this file; `b` is the dense (n_cells, <gene-pair-axis width>) operand -- the
            # exact tensor CU-E is supposed to bound to (n_cells, chunk) / (n_cells, |union|).
            _record(b)
            out = orig_sparse_mm(a, b, *args, **kwargs)
            _record(out)
            return out

        cclm.torch.zeros = spy_zeros
        cclm.torch.zeros_like = spy_zeros_like
        cclm.torch.sparse.mm = spy_sparse_mm
        try:
            compute_interacting_cell_scores_lowmem(
                adata, **{**CALL_KWARGS, "gene_pair_chunk_size": gp_chunk, "metabolite_chunk_size": m_chunk}
            )
        finally:
            cclm.torch.zeros = orig_zeros
            cclm.torch.zeros_like = orig_zeros_like
            cclm.torch.sparse.mm = orig_sparse_mm

        self.assertTrue(
            seen_widths,
            "no 2D tensors observed through torch.zeros/zeros_like/sparse.mm -- test is vacuous",
        )
        return max(seen_widths)

    def test_chunked_run_is_bounded_and_unchunked_run_reaches_full_width(self):
        n_gp, n_m = 4, 3  # make_shared_pair_test_adata
        # Largest legitimate width at chunk size 1 is bounded by the widest single
        # metabolite's own gene-pair union (Metab_A and Metab_C each own 2 pairs here) --
        # strictly less than n_gp=4, the old unchunked width this guard checks for.
        max_allowed_width_chunked = 2

        max_width_chunked = self._measure_max_gp_axis_width(
            make_shared_pair_test_adata(seed=23), gp_chunk=1, m_chunk=1
        )
        self.assertLessEqual(
            max_width_chunked, max_allowed_width_chunked,
            f"a gene-pair-axis tensor of width {max_width_chunked} passed through "
            f"torch.zeros/zeros_like/sparse.mm despite chunk sizes of 1 (> "
            f"{max_allowed_width_chunked}) -- chunking failed to bound GPU memory as claimed.",
        )

        # POSITIVE CONTROL: same measurement, unchunked (chunk sizes >= n_gp/n_m) -- must
        # reach the FULL n_gp width somewhere (Pass 1's single, full-size chunk), proving
        # this metric can detect an unbounded allocation and the bounded result above is
        # not vacuous.
        max_width_unchunked = self._measure_max_gp_axis_width(
            make_shared_pair_test_adata(seed=23), gp_chunk=n_gp, m_chunk=n_m
        )
        self.assertEqual(
            max_width_unchunked, n_gp,
            f"expected the unchunked run to reach the full gene-pair width n_gp={n_gp} "
            f"somewhere (Pass 1's single chunk) -- got {max_width_unchunked}; either the "
            f"fixture changed or this measurement no longer reflects the gene-pair axis.",
        )
        self.assertLess(
            max_width_chunked, max_width_unchunked,
            "the chunked and unchunked runs measured the same max width -- this guard "
            "would not have caught a regression back to full-width allocation.",
        )


if __name__ == "__main__":
    unittest.main()
