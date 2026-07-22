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
import cell_communication_lowmem
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


def make_heavy_sharing_test_adata(n_metabolites=10, n_cells=N_CELLS, seed=0):
    """CU-F hardening fixture: MANY metabolites (default 10) that all reference the SAME
    single gene pair. The gene-pair UNION of any subset of these metabolites is always
    size 1 (they all point at the one pair) -- so a chunker that bounds ONLY the union
    (the pre-hardening `_greedy_metabolite_chunks` behavior when `metabolite_chunk_size=
    None`) would put ALL `n_metabolites` metabolites in ONE chunk regardless of budget,
    producing an unbounded `(n_cells, n_metabolites)` tensor. This is exactly the
    metabolite-COUNT-axis gap the implicit `metab_count_cap` hardening closes: with it,
    a chunk also closes at `max_pairs_per_chunk` metabolites even though the union never
    forces it to.
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

    adata.uns["gene_pairs"] = [["G1", "G2"]]  # the ONE shared pair, index 0

    metab_names = [f"HeavyM{i}" for i in range(n_metabolites)]
    adata.uns["gene_pairs_per_metabolite"] = {
        name: {
            "gene_pair": [["G1", "G2"]],
            "gene_type": [["IMP-EXP", "IMP-EXP"]],
        }
        for name in metab_names
    }
    adata.uns["ccc_results"] = {
        "cell_com_df_gp_sig": pd.DataFrame({"Gene 1": ["G1"], "Gene 2": ["G2"]}),
        "cell_com_df_m_sig": pd.DataFrame({"Metabolite": metab_names}),
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

    def test_tiny_element_budget_union_sizing_matches_stock(self):
        """CU-F: with a TINY ``element_budget`` (small enough that ``max_pairs_per_chunk``
        is 1 or 2), Pass 2's GREEDY union-based grouping (``_greedy_metabolite_chunks``)
        must produce MULTIPLE metabolite chunks whose unions span/share gene pairs -- and
        the result must still be bit-for-bit identical to stock. This is the key proof
        that variable-width union sizing (the fix for the real Savio Pass-2 OOM) doesn't
        break exactness. Uses both many-to-many fixtures, incl. `make_many_pairs_test_
        adata` (a >=3-pair metabolite, so a lone-metabolite-exceeds-budget chunk is also
        exercised)."""
        for factory in (make_shared_pair_test_adata, make_many_pairs_test_adata):
            for element_budget in (1 * N_CELLS, 2 * N_CELLS):  # -> max_pairs_per_chunk = 1 or 2
                with self.subTest(factory=factory.__name__, element_budget=element_budget):
                    adata_stock = factory(seed=37)
                    adata_lowmem = factory(seed=37)

                    harreman.tools.compute_interacting_cell_scores(adata_stock, **CALL_KWARGS)
                    lowmem_kwargs = dict(CALL_KWARGS)
                    lowmem_kwargs["element_budget"] = element_budget
                    compute_interacting_cell_scores_lowmem(adata_lowmem, **lowmem_kwargs)

                    stock_np = adata_stock.uns["interacting_cell_results"]["np"]
                    lowmem_np = adata_lowmem.uns["interacting_cell_results"]["np"]
                    for grain in ("gp", "m"):
                        for key in ("cs", "pval", "FDR", "cs_sig_pval", "cs_sig_FDR"):
                            np.testing.assert_array_equal(
                                stock_np[grain][key], lowmem_np[grain][key],
                                err_msg=(
                                    f"mismatch in ['np']['{grain}']['{key}'] "
                                    f"(factory={factory.__name__}, element_budget={element_budget})"
                                ),
                            )

    def test_heavy_sharing_metabolite_count_axis_matches_stock(self):
        """CU-F hardening (review pass 2): `make_heavy_sharing_test_adata` has 10
        metabolites all referencing the SAME single gene pair, so the gene-pair UNION never
        forces a chunk split -- only the metabolite-COUNT cap does. This is the fixture
        that specifically exercises the implicit `metab_count_cap` (not just the union
        budget already covered above), and the result must still be bit-for-bit identical
        to stock."""
        for element_budget in (1 * N_CELLS, 2 * N_CELLS, 3 * N_CELLS):
            with self.subTest(element_budget=element_budget):
                adata_stock = make_heavy_sharing_test_adata(seed=43)
                adata_lowmem = make_heavy_sharing_test_adata(seed=43)

                harreman.tools.compute_interacting_cell_scores(adata_stock, **CALL_KWARGS)
                lowmem_kwargs = dict(CALL_KWARGS)
                lowmem_kwargs["element_budget"] = element_budget
                compute_interacting_cell_scores_lowmem(adata_lowmem, **lowmem_kwargs)

                stock_np = adata_stock.uns["interacting_cell_results"]["np"]
                lowmem_np = adata_lowmem.uns["interacting_cell_results"]["np"]
                for grain in ("gp", "m"):
                    for key in ("cs", "pval", "FDR", "cs_sig_pval", "cs_sig_FDR"):
                        np.testing.assert_array_equal(
                            stock_np[grain][key], lowmem_np[grain][key],
                            err_msg=(
                                f"mismatch in ['np']['{grain}']['{key}'] "
                                f"(heavy_sharing, element_budget={element_budget})"
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

    def test_pass2_bounded_by_element_budget_alone_no_metabolite_chunk_size(self):
        """CU-F: guards the actual bug being fixed -- Pass 2 must be bounded by the
        gene-pair-UNION footprint driven by ``element_budget``, even when ``metabolite_
        chunk_size`` is left at its fully-automatic default (``None``). Before this fix,
        Pass 2 was sized by metabolite COUNT alone (``element_budget // n_cells``
        metabolites per chunk), which for a many-to-many gene-pair<->metabolite map could
        reference nearly the full ``n_gp`` -- silently rebuilding the very
        ``(n_cells, n_gp)``-scale tensors Pass 1 chunks to avoid (the real Savio OOM this
        retrofit fixes). Uses ``make_many_pairs_test_adata`` (Metab_C owns 3 gene pairs)
        with a tiny ``element_budget`` so ``max_pairs_per_chunk`` = 2 -- Metab_C's own
        union (3) must still get its own over-budget chunk (a metabolite's pairs can't be
        split), but no OTHER ``torch.sparse.mm`` operand/output should exceed 2."""
        adata = make_many_pairs_test_adata(seed=41)
        element_budget = 2 * N_CELLS  # -> max_pairs_per_chunk = 2
        max_allowed_width = 3  # Metab_C's own union size -- the one allowed exception

        seen_widths = []

        def _record(t):
            if hasattr(t, "dim") and t.dim() == 2:
                seen_widths.append(t.shape[1])

        orig_sparse_mm = cell_communication_lowmem.torch.sparse.mm

        def spy_sparse_mm(a, b, *args, **kwargs):
            _record(b)
            out = orig_sparse_mm(a, b, *args, **kwargs)
            _record(out)
            return out

        cell_communication_lowmem.torch.sparse.mm = spy_sparse_mm
        try:
            compute_interacting_cell_scores_lowmem(
                adata, **{**CALL_KWARGS, "element_budget": element_budget, "metabolite_chunk_size": None}
            )
        finally:
            cell_communication_lowmem.torch.sparse.mm = orig_sparse_mm

        self.assertTrue(seen_widths, "no 2D tensors observed through torch.sparse.mm -- test is vacuous")
        self.assertLessEqual(
            max(seen_widths), max_allowed_width,
            f"a tensor of width {max(seen_widths)} passed through torch.sparse.mm despite "
            f"element_budget={element_budget} (max_pairs_per_chunk=2) and metabolite_chunk_"
            f"size=None -- Pass 2 is not bounded by the gene-pair-union footprint as claimed "
            f"(this is exactly the Savio Pass-2 OOM this test guards against).",
        )

    def test_metabolite_count_axis_is_bounded_by_element_budget(self):
        """CU-F hardening (review pass 2): for `make_heavy_sharing_test_adata` (all
        metabolites share ONE gene pair, so the gene-pair-UNION axis never forces a chunk
        split), the metabolite-COUNT axis (`x_m_*_c`, shape `(n_cells, |metabs in
        chunk|)`, allocated via `torch.zeros`) must still be bounded by `max_pairs_per_
        chunk` -- this is what the implicit `metab_count_cap` hardening actually guards
        against (the union-only bound above is provably insufficient on this fixture).
        Includes a POSITIVE CONTROL: the default (huge) `element_budget` must reach the
        full metabolite count (all 10 in one chunk), proving the bounded result at a tiny
        budget isn't vacuous."""
        n_metabolites = 10

        def _measure(element_budget):
            adata = make_heavy_sharing_test_adata(n_metabolites=n_metabolites, seed=47)
            seen_widths = []

            def _record(t):
                if hasattr(t, "dim") and t.dim() == 2:
                    seen_widths.append(t.shape[1])

            orig_zeros = cell_communication_lowmem.torch.zeros
            orig_zeros_like = cell_communication_lowmem.torch.zeros_like

            def spy_zeros(*args, **kwargs):
                t = orig_zeros(*args, **kwargs)
                _record(t)
                return t

            def spy_zeros_like(*args, **kwargs):
                t = orig_zeros_like(*args, **kwargs)
                _record(t)
                return t

            cell_communication_lowmem.torch.zeros = spy_zeros
            cell_communication_lowmem.torch.zeros_like = spy_zeros_like
            try:
                compute_interacting_cell_scores_lowmem(adata, **{**CALL_KWARGS, "element_budget": element_budget})
            finally:
                cell_communication_lowmem.torch.zeros = orig_zeros
                cell_communication_lowmem.torch.zeros_like = orig_zeros_like

            self.assertTrue(seen_widths, "no 2D torch.zeros/zeros_like tensors observed -- test is vacuous")
            return max(seen_widths)

        tiny_budget = 2 * N_CELLS  # -> max_pairs_per_chunk = 2
        max_width_bounded = _measure(tiny_budget)
        self.assertLessEqual(
            max_width_bounded, 2,
            f"a torch.zeros/zeros_like tensor of width {max_width_bounded} was allocated "
            f"despite element_budget={tiny_budget} (max_pairs_per_chunk=2) -- the "
            f"metabolite-COUNT axis is not bounded (this is exactly the gap the implicit "
            f"metab_count_cap hardening closes: heavy sharing means the gene-pair-UNION "
            f"axis alone never forces a chunk split).",
        )

        # POSITIVE CONTROL: default (huge) element_budget -> all n_metabolites in one chunk.
        max_width_unbounded = _measure(50_000_000)
        self.assertEqual(
            max_width_unbounded, n_metabolites,
            f"expected the default-budget run to reach the full metabolite count "
            f"n_metabolites={n_metabolites} somewhere -- got {max_width_unbounded}; either "
            f"the fixture changed or this measurement no longer reflects the metabolite-"
            f"count axis (i.e. the bounded result above would not be proven non-vacuous).",
        )

    def test_explicit_gene_pair_chunk_size_is_clamped_by_element_budget(self):
        """CU-F hardening (review point 3): a PINNED ``gene_pair_chunk_size`` larger than
        what a tiny ``element_budget`` would allow must still be CLAMPED down to
        ``max_pairs_per_chunk`` (``min(explicit, max_pairs_per_chunk)``), not used
        verbatim -- otherwise a caller pinning a chunk size that itself OOMs would re-OOM
        identically on every retry, since only ``element_budget`` shrinks across attempts.
        Uses ``make_many_pairs_test_adata`` (n_gp=5) with a huge explicit
        ``gene_pair_chunk_size`` and a tiny ``element_budget``."""
        adata = make_many_pairs_test_adata(seed=53)
        element_budget = 2 * N_CELLS  # -> max_pairs_per_chunk = 2
        huge_explicit_gp_chunk = 1000  # >> n_gp=5 and >> max_pairs_per_chunk=2

        seen_widths = []

        def _record(t):
            if hasattr(t, "dim") and t.dim() == 2:
                seen_widths.append(t.shape[1])

        orig_sparse_mm = cell_communication_lowmem.torch.sparse.mm

        def spy_sparse_mm(a, b, *args, **kwargs):
            _record(b)
            out = orig_sparse_mm(a, b, *args, **kwargs)
            _record(out)
            return out

        cell_communication_lowmem.torch.sparse.mm = spy_sparse_mm
        try:
            compute_interacting_cell_scores_lowmem(
                adata, **{
                    **CALL_KWARGS,
                    "element_budget": element_budget,
                    "gene_pair_chunk_size": huge_explicit_gp_chunk,
                }
            )
        finally:
            cell_communication_lowmem.torch.sparse.mm = orig_sparse_mm

        self.assertTrue(seen_widths, "no 2D tensors observed -- test is vacuous")
        # Metab_C's own union (3) is the one allowed exception (Pass 2); Pass 1's
        # gene-pair chunk width should be clamped to max_pairs_per_chunk=2, not the
        # pinned 1000.
        self.assertLessEqual(
            max(seen_widths), 3,
            f"a tensor of width {max(seen_widths)} passed through torch.sparse.mm despite "
            f"a tiny element_budget={element_budget} (max_pairs_per_chunk=2) -- the pinned "
            f"gene_pair_chunk_size={huge_explicit_gp_chunk} was used verbatim instead of "
            f"being clamped to the budget, defeating OOM-halving on the override path.",
        )


class GreedyMetaboliteChunkingUnitTests(unittest.TestCase):
    """Deterministic, known-answer unit tests for
    ``cell_communication_lowmem._greedy_metabolite_chunks`` in isolation (no adata, no
    torch) -- the CU-F algorithm that replaced fixed-metabolite-count Pass-2 chunking with
    gene-pair-UNION-footprint-bounded chunking."""

    def setUp(self):
        # M1: 1 pair; M2: 1 pair (shares nothing with M1); M3: 1 pair; M4: a WIDE
        # metabolite with 8 pairs (like a real large metabolite, 05_harreman_reference.md
        # sec.4a); M5: 1 pair. Chosen so a budget of 2 both GROUPS some metabolites
        # together (M1+M2, whose union stays <= budget) AND forces a lone, over-budget
        # chunk for M4 (whose own union already exceeds the budget).
        self.metabolites = ["M1", "M2", "M3", "M4", "M5"]
        self.gene_pair_dict = {
            "M1": [0],
            "M2": [1],
            "M3": [2],
            "M4": [3, 4, 5, 6, 7, 8, 9, 10],
            "M5": [11],
        }

    def _chunks(self, max_pairs_per_chunk, hard_cap=None):
        return list(cell_communication_lowmem._greedy_metabolite_chunks(
            self.metabolites, self.gene_pair_dict, max_pairs_per_chunk, hard_cap
        ))

    def test_budget_2_groups_small_metabolites_and_isolates_the_wide_one(self):
        chunks = self._chunks(max_pairs_per_chunk=2)
        grouped = [self.metabolites[sl] for sl in chunks]
        self.assertEqual(grouped, [["M1", "M2"], ["M3"], ["M4"], ["M5"]])

        for metabs in grouped:
            union = set().union(*(set(self.gene_pair_dict[m]) for m in metabs))
            if len(metabs) == 1 and metabs[0] == "M4":
                # the one allowed exception: a lone metabolite whose own pairs already
                # exceed the budget still gets its own (over-budget) chunk.
                self.assertEqual(len(union), 8)
                self.assertGreater(len(union), 2)
            else:
                self.assertLessEqual(
                    len(union), 2,
                    f"chunk {metabs} has union size {len(union)} > max_pairs_per_chunk=2",
                )

    def test_coverage_and_order_preserved(self):
        """Every metabolite appears in EXACTLY ONE chunk, in original order, regardless of
        budget -- moving chunk boundaries must never drop or reorder a metabolite."""
        for max_pairs_per_chunk in (1, 2, 3, 100):
            with self.subTest(max_pairs_per_chunk=max_pairs_per_chunk):
                chunks = self._chunks(max_pairs_per_chunk)
                covered = [m for sl in chunks for m in self.metabolites[sl]]
                self.assertEqual(covered, self.metabolites)

    def test_budget_1_isolates_every_metabolite(self):
        chunks = self._chunks(max_pairs_per_chunk=1)
        grouped = [self.metabolites[sl] for sl in chunks]
        self.assertEqual(grouped, [["M1"], ["M2"], ["M3"], ["M4"], ["M5"]])

    def test_large_budget_is_a_single_chunk(self):
        chunks = self._chunks(max_pairs_per_chunk=1_000_000)
        grouped = [self.metabolites[sl] for sl in chunks]
        self.assertEqual(grouped, [self.metabolites])

    def test_hard_cap_closes_chunks_even_under_budget(self):
        """``hard_cap`` (``metabolite_chunk_size``) closes a chunk at N metabolites even
        when the union budget alone would have allowed more -- "whichever limit hits
        first"."""
        chunks = self._chunks(max_pairs_per_chunk=1_000_000, hard_cap=1)
        grouped = [self.metabolites[sl] for sl in chunks]
        self.assertEqual(grouped, [["M1"], ["M2"], ["M3"], ["M4"], ["M5"]])

        chunks2 = self._chunks(max_pairs_per_chunk=1_000_000, hard_cap=2)
        grouped2 = [self.metabolites[sl] for sl in chunks2]
        self.assertEqual(grouped2, [["M1", "M2"], ["M3", "M4"], ["M5"]])


class CellCommunicationLowmemOOMFallbackTests(unittest.TestCase):
    """CU-F automatic OOM-halving retry, exercised WITHOUT real CUDA: injects a fake
    CUDA-OOM-style ``RuntimeError`` on the very first ``torch.sparse.mm`` call (i.e. the
    start of attempt 1's Pass 1), then lets every later call proceed normally --
    simulating "OOM on attempt 1, succeeds on attempt 2 at the halved budget". Also
    monkeypatches ``torch.cuda.is_available``/``torch.cuda.empty_cache`` (this test
    machine has no real CUDA) so the empty-cache-on-OOM branch is actually exercised and
    observable, per the review's request."""

    def test_oom_is_caught_budget_halved_and_retry_succeeds(self):
        import io
        import contextlib

        # Reference: the un-failed path (no injected OOM, default element_budget=50M).
        adata_baseline = make_many_pairs_test_adata(seed=29)
        compute_interacting_cell_scores_lowmem(adata_baseline, **CALL_KWARGS)
        baseline_np = adata_baseline.uns["interacting_cell_results"]["np"]

        adata = make_many_pairs_test_adata(seed=29)

        orig_sparse_mm = cell_communication_lowmem.torch.sparse.mm
        orig_is_available = cell_communication_lowmem.torch.cuda.is_available
        orig_empty_cache = cell_communication_lowmem.torch.cuda.empty_cache
        call_count = {"n": 0}
        empty_cache_calls = {"n": 0}

        def flaky_sparse_mm(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("CUDA out of memory. Tried to allocate 9.00 GiB")
            return orig_sparse_mm(*args, **kwargs)

        def spy_empty_cache():
            empty_cache_calls["n"] += 1

        cell_communication_lowmem.torch.sparse.mm = flaky_sparse_mm
        cell_communication_lowmem.torch.cuda.is_available = lambda: True
        cell_communication_lowmem.torch.cuda.empty_cache = spy_empty_cache
        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                compute_interacting_cell_scores_lowmem(adata, **CALL_KWARGS)
        finally:
            cell_communication_lowmem.torch.sparse.mm = orig_sparse_mm
            cell_communication_lowmem.torch.cuda.is_available = orig_is_available
            cell_communication_lowmem.torch.cuda.empty_cache = orig_empty_cache

        printed = stdout.getvalue()
        self.assertIn("CUDA OOM at element_budget=50000000", printed)
        self.assertIn("retrying at element_budget=25000000", printed)
        # >=1, not ==1: `torch.cuda.empty_cache()` is ALSO called unconditionally between
        # Pass 1 and Pass 2 on every (including successful) attempt (CU-F inter-pass
        # cleanup) -- so a run with exactly one retry legitimately calls it twice (once
        # from the caught-OOM handler, once from attempt 2's own inter-pass cleanup once
        # it reaches Pass 2). The property under test is just "the OOM handler really
        # calls empty_cache", which >=1 already proves without over-specifying the total.
        self.assertGreaterEqual(empty_cache_calls["n"], 1, "torch.cuda.empty_cache() was never called on the caught OOM")
        self.assertGreater(call_count["n"], 1, "torch.sparse.mm was never called again after the injected OOM -- no retry happened")

        lowmem_np = adata.uns["interacting_cell_results"]["np"]
        for grain in ("gp", "m"):
            for key in ("cs", "pval", "FDR", "cs_sig_pval", "cs_sig_FDR"):
                np.testing.assert_array_equal(
                    baseline_np[grain][key], lowmem_np[grain][key],
                    err_msg=f"post-OOM-retry result diverged from the un-failed baseline in ['np']['{grain}']['{key}']",
                )

    def test_oom_persists_past_max_retries_reraises(self):
        """If every attempt OOMs, the error is re-raised after `max_oom_retries` attempts
        (not silently swallowed)."""
        def always_oom(*args, **kwargs):
            raise RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB")

        adata = make_shared_pair_test_adata(seed=31)
        orig_sparse_mm = cell_communication_lowmem.torch.sparse.mm
        cell_communication_lowmem.torch.sparse.mm = always_oom
        try:
            with self.assertRaises(RuntimeError) as ctx:
                compute_interacting_cell_scores_lowmem(adata, **{**CALL_KWARGS, "max_oom_retries": 2})
            self.assertIn("out of memory", str(ctx.exception).lower())
        finally:
            cell_communication_lowmem.torch.sparse.mm = orig_sparse_mm

    def test_non_oom_exception_is_not_caught(self):
        """A non-OOM RuntimeError from inside the branch must propagate immediately on the
        FIRST attempt, not be treated as a retryable OOM."""
        def broken_sparse_mm(*args, **kwargs):
            raise RuntimeError("some unrelated failure, not an OOM")

        adata = make_shared_pair_test_adata(seed=31)
        orig_sparse_mm = cell_communication_lowmem.torch.sparse.mm
        cell_communication_lowmem.torch.sparse.mm = broken_sparse_mm
        try:
            with self.assertRaises(RuntimeError) as ctx:
                compute_interacting_cell_scores_lowmem(adata, **CALL_KWARGS)
            self.assertIn("unrelated failure", str(ctx.exception))
        finally:
            cell_communication_lowmem.torch.sparse.mm = orig_sparse_mm

    def test_max_oom_retries_zero_or_one_still_produces_populated_result(self):
        """Review hardening (point 1): `max_oom_retries=0` (or 1) must NOT mean "skip the
        branch" -- `range(max_oom_retries)` alone would run zero times, silently leaving
        `uns['interacting_cell_results']['np']` unpopulated (a KeyError far downstream in
        `summarize_nbhd_scores`, not a loud failure here). With no injected OOM, both
        values must produce a fully populated result, bit-identical to the default-retries
        run."""
        adata_baseline = make_shared_pair_test_adata(seed=59)
        compute_interacting_cell_scores_lowmem(adata_baseline, **CALL_KWARGS)
        baseline_np = adata_baseline.uns["interacting_cell_results"]["np"]

        for max_oom_retries in (0, 1):
            with self.subTest(max_oom_retries=max_oom_retries):
                adata = make_shared_pair_test_adata(seed=59)
                compute_interacting_cell_scores_lowmem(
                    adata, **{**CALL_KWARGS, "max_oom_retries": max_oom_retries}
                )
                self.assertIn("interacting_cell_results", adata.uns)
                self.assertIn("np", adata.uns["interacting_cell_results"])
                np_res = adata.uns["interacting_cell_results"]["np"]
                for grain in ("gp", "m"):
                    self.assertIn("cs", np_res[grain])
                    self.assertIn("pval", np_res[grain])
                    for key in ("cs", "pval", "FDR", "cs_sig_pval", "cs_sig_FDR"):
                        np.testing.assert_array_equal(
                            baseline_np[grain][key], np_res[grain][key],
                            err_msg=f"mismatch in ['np']['{grain}']['{key}'] (max_oom_retries={max_oom_retries})",
                        )

    def test_element_budget_none_direct_call_matches_default(self):
        """Review hardening (point 4, nit): a caller passing `element_budget=None`
        directly must not `TypeError` at `cur_element_budget // n_cells` -- it should
        behave exactly like the signature's own default (50,000,000)."""
        adata_baseline = make_shared_pair_test_adata(seed=61)
        compute_interacting_cell_scores_lowmem(adata_baseline, **CALL_KWARGS)
        baseline_np = adata_baseline.uns["interacting_cell_results"]["np"]

        adata = make_shared_pair_test_adata(seed=61)
        compute_interacting_cell_scores_lowmem(adata, **{**CALL_KWARGS, "element_budget": None})
        np_res = adata.uns["interacting_cell_results"]["np"]
        for grain in ("gp", "m"):
            for key in ("cs", "pval", "FDR", "cs_sig_pval", "cs_sig_FDR"):
                np.testing.assert_array_equal(
                    baseline_np[grain][key], np_res[grain][key],
                    err_msg=f"mismatch in ['np']['{grain}']['{key}'] (element_budget=None)",
                )

    def test_oom_first_surfaces_in_pass2_is_caught_and_retried(self):
        """Review test gap (point 5): the earlier OOM-fallback test only injects a
        failure in Pass 1 (the very first `torch.sparse.mm` call). This test injects the
        failure into `compute_metabolite_cs`, which Pass 1 NEVER calls (only Pass 2's
        observed-score and per-permutation metabolite reductions do) -- so this
        specifically simulates "Pass 1 completes fine, Pass 2 OOMs", and confirms it is
        still caught, retried, and produces a bit-identical result."""
        adata_baseline = make_many_pairs_test_adata(seed=67)
        compute_interacting_cell_scores_lowmem(adata_baseline, **CALL_KWARGS)
        baseline_np = adata_baseline.uns["interacting_cell_results"]["np"]

        adata = make_many_pairs_test_adata(seed=67)
        orig_compute_metabolite_cs = cell_communication_lowmem.compute_metabolite_cs
        call_count = {"n": 0}

        def flaky_compute_metabolite_cs(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("CUDA out of memory. Tried to allocate 9.00 GiB")
            return orig_compute_metabolite_cs(*args, **kwargs)

        cell_communication_lowmem.compute_metabolite_cs = flaky_compute_metabolite_cs
        try:
            compute_interacting_cell_scores_lowmem(adata, **CALL_KWARGS)
        finally:
            cell_communication_lowmem.compute_metabolite_cs = orig_compute_metabolite_cs

        self.assertGreater(call_count["n"], 1, "compute_metabolite_cs was never called again after the injected OOM -- no retry happened")
        np_res = adata.uns["interacting_cell_results"]["np"]
        for grain in ("gp", "m"):
            for key in ("cs", "pval", "FDR", "cs_sig_pval", "cs_sig_FDR"):
                np.testing.assert_array_equal(
                    baseline_np[grain][key], np_res[grain][key],
                    err_msg=f"post-Pass-2-OOM-retry result diverged from baseline in ['np']['{grain}']['{key}']",
                )

    def test_multiple_halvings_before_success(self):
        """Review test gap (point 5): injects a failure on the first TWO attempts'
        opening `torch.sparse.mm` calls (each retry restarts `_run_non_parametric_pass`
        from scratch, so its first `torch.sparse.mm` call is the next global call), so
        attempt 3 (budget halved TWICE: 50M -> 25M -> 12.5M) is what finally succeeds --
        and confirms the result is still bit-identical to the un-failed baseline."""
        adata_baseline = make_shared_pair_test_adata(seed=71)
        compute_interacting_cell_scores_lowmem(adata_baseline, **CALL_KWARGS)
        baseline_np = adata_baseline.uns["interacting_cell_results"]["np"]

        adata = make_shared_pair_test_adata(seed=71)
        orig_sparse_mm = cell_communication_lowmem.torch.sparse.mm
        call_count = {"n": 0}

        def flaky_sparse_mm(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                raise RuntimeError("CUDA out of memory. Tried to allocate 9.00 GiB")
            return orig_sparse_mm(*args, **kwargs)

        cell_communication_lowmem.torch.sparse.mm = flaky_sparse_mm
        try:
            compute_interacting_cell_scores_lowmem(adata, **{**CALL_KWARGS, "max_oom_retries": 4})
        finally:
            cell_communication_lowmem.torch.sparse.mm = orig_sparse_mm

        self.assertGreater(call_count["n"], 2, "expected at least 3 attempts (2 failures + 1 success)")
        np_res = adata.uns["interacting_cell_results"]["np"]
        for grain in ("gp", "m"):
            for key in ("cs", "pval", "FDR", "cs_sig_pval", "cs_sig_FDR"):
                np.testing.assert_array_equal(
                    baseline_np[grain][key], np_res[grain][key],
                    err_msg=f"post-multi-halving-retry result diverged from baseline in ['np']['{grain}']['{key}']",
                )


if __name__ == "__main__":
    unittest.main()
