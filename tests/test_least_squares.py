"""Tier-1 tests for `metab_processing/LinearRegression/least_squares.py`.

Builds a synthetic AnnData directly against the store contract that `build_x.py` produces
(no estimator/torch involved -- `get_gene_factors` is pure pandas), with a KNOWN linear
generative model so OLS must recover the true betas. Rev 5: the fixture carries BOTH
sources ('imputed' and 'lognorm', see `build_x.SOURCE_LAYERS`), each with its OWN factor
block, metab column, known betas, and y layer (`imputed_count` / `normalized_count`) --
deliberately distinct column names and values between sources so a test recovering the
wrong source's betas, or reading the wrong layer, would actually fail rather than pass by
coincidence.

Rev 5.1 (backward compat): `source='imputed'` uses the ORIGINAL unsuffixed keys
(`x_factors`, `x_factors_cols`, `x_factor_map`, `x_metab`, `x_metab_modulators`,
`x_genes`) -- byte-identical to the pre-dual-block scheme, so an existing x_adata.h5ad
stays readable. Only `source='lognorm'` is `_lognorm`-suffixed. See `build_x._SOURCE_SUFFIX`.

Fixture: 40 cells, an obs['ct'] annotation ('T'/'B', 20 each), two genes:
  - 'G' uses all 3 factor columns of its source plus that source's shared metab column;
    y_G is an EXACT linear combination of all four (known betas) plus an intercept -- no
    noise, so OLS recovery is exact to floating precision.
  - 'H' uses only the source's first TF factor column; y_H depends only on that column, so
    when 'all' metabolites are requested the extra metab column has a true coefficient of 0
    (tests OLS handling an unused predictor).
"""
import os
import sys
import unittest
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from anndata import AnnData

from metab_processing.SpaceTravLR.beta_analysis import _group
from metab_processing.LinearRegression.build_x import get_gene_factors
from metab_processing.LinearRegression.least_squares import (
    fit_gene_betas,
    _fit,
    _fit_ols,
    _select_cells,
    subsample_cells,
    subsample_gene_betas,
    rank_coefficients,
    plot_top_coefficients,
    plot_beta_histogram,
)

N = 40

FACTOR_COLS_IMPUTED = ["TF_A", "TF_B", "L$R"]
METAB_COL_IMPUTED = "metab@Glucose"
KNOWN_BETA_G_IMPUTED = {"TF_A": 1.5, "TF_B": -2.0, "L$R": 0.7, METAB_COL_IMPUTED: 0.9}
INTERCEPT_G_IMPUTED = 0.3
KNOWN_BETA_H_IMPUTED = {"TF_A": -1.2}
INTERCEPT_H_IMPUTED = 0.05

# 'lognorm' fixture: deliberately DIFFERENT column names and coefficients from 'imputed'
# so a bug that reads the wrong source's block/layer produces a visibly wrong (not
# coincidentally-correct) recovery.
FACTOR_COLS_LOGNORM = ["TF_A2", "TF_B2", "L$R2"]
METAB_COL_LOGNORM = "metab@Fructose"
KNOWN_BETA_G_LOGNORM = {"TF_A2": 2.2, "TF_B2": -0.4, "L$R2": 1.1, METAB_COL_LOGNORM: -0.6}
INTERCEPT_G_LOGNORM = -0.15
KNOWN_BETA_H_LOGNORM = {"TF_A2": 0.85}
INTERCEPT_H_LOGNORM = 0.2


def _make_adata():
    rng = np.random.default_rng(0)
    factors_imp = rng.normal(size=(N, 3))   # columns: TF_A, TF_B, L$R
    metab_imp = rng.normal(size=(N, 1))     # column: metab@Glucose

    rng2 = np.random.default_rng(42)
    factors_log = rng2.normal(size=(N, 3))  # columns: TF_A2, TF_B2, L$R2
    metab_log = rng2.normal(size=(N, 1))    # column: metab@Fructose

    y_g_imp = (factors_imp @ np.array([KNOWN_BETA_G_IMPUTED[c] for c in FACTOR_COLS_IMPUTED])
               + metab_imp[:, 0] * KNOWN_BETA_G_IMPUTED[METAB_COL_IMPUTED] + INTERCEPT_G_IMPUTED)
    y_h_imp = factors_imp[:, 0] * KNOWN_BETA_H_IMPUTED["TF_A"] + INTERCEPT_H_IMPUTED

    y_g_log = (factors_log @ np.array([KNOWN_BETA_G_LOGNORM[c] for c in FACTOR_COLS_LOGNORM])
               + metab_log[:, 0] * KNOWN_BETA_G_LOGNORM[METAB_COL_LOGNORM] + INTERCEPT_G_LOGNORM)
    y_h_log = factors_log[:, 0] * KNOWN_BETA_H_LOGNORM["TF_A2"] + INTERCEPT_H_LOGNORM

    obs_names = [f"c{i}" for i in range(N)]
    adata = AnnData(X=np.zeros((N, 2), dtype=np.float64))
    adata.var_names = ["G", "H"]
    adata.obs_names = obs_names
    adata.obs["ct"] = ["T"] * (N // 2) + ["B"] * (N // 2)
    adata.layers["imputed_count"] = np.column_stack([y_g_imp, y_h_imp])
    adata.layers["normalized_count"] = np.column_stack([y_g_log, y_h_log])

    # 'imputed' uses the ORIGINAL unsuffixed keys (backward compat with pre-dual-block
    # x_adata.h5ad files); only 'lognorm' is suffixed.
    adata.obsm["x_factors"] = factors_imp
    adata.uns["x_factors_cols"] = list(FACTOR_COLS_IMPUTED)
    adata.uns["x_factor_map"] = {"G": list(FACTOR_COLS_IMPUTED), "H": ["TF_A"]}
    adata.obsm["x_metab"] = metab_imp
    adata.uns["x_metab_modulators"] = [METAB_COL_IMPUTED]
    adata.uns["x_genes"] = ["G", "H"]

    adata.obsm["x_factors_lognorm"] = factors_log
    adata.uns["x_factors_lognorm_cols"] = list(FACTOR_COLS_LOGNORM)
    adata.uns["x_factor_map_lognorm"] = {"G": list(FACTOR_COLS_LOGNORM), "H": ["TF_A2"]}
    adata.obsm["x_metab_lognorm"] = metab_log
    adata.uns["x_metab_lognorm_modulators"] = [METAB_COL_LOGNORM]
    adata.uns["x_genes_lognorm"] = ["G", "H"]

    return adata


class FitOlsTests(unittest.TestCase):
    def test_recovers_known_coefficients_with_intercept(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(50, 3))
        beta_true = np.array([2.0, -1.0, 0.5])
        y = X @ beta_true + 0.75
        beta, r2 = _fit_ols(X, y)
        np.testing.assert_allclose(beta, beta_true, atol=1e-8)
        self.assertAlmostEqual(r2, 1.0, places=8)

    def test_r2_zero_when_ss_tot_zero(self):
        X = np.zeros((5, 2))
        y = np.full(5, 3.0)
        beta, r2 = _fit_ols(X, y)
        self.assertEqual(r2, 0.0)

    def test_collinear_columns_min_norm_not_a_correctness_claim(self):
        """X's first two columns are exactly collinear (x1, 2*x1); y depends only on x1
        with true coefficient 3 (e.g. beta=[3,0,0] or beta=[0,1.5,0] would both fit
        perfectly). `lstsq` returns the MIN-NORM solution among these, so the fit
        (X @ beta ~= y, r2 ~= 1) is still correct but the SPLIT across the two
        collinear columns is not [3, 0] -- coefficients are non-unique here, not wrong."""
        rng = np.random.default_rng(7)
        x1 = rng.normal(size=60)
        noise_col = rng.normal(size=60)
        X = np.column_stack([x1, 2 * x1, noise_col])
        y = 3 * x1
        beta, r2 = _fit_ols(X, y)
        self.assertAlmostEqual(r2, 1.0, places=6)
        np.testing.assert_allclose(X @ beta, y, atol=1e-6)
        # documents non-uniqueness: min-norm solution spreads weight across the
        # collinear pair rather than loading it all onto column 0.
        self.assertFalse(np.allclose(beta[:2], [3.0, 0.0], atol=1e-3))


class FitGeneBetasTests(unittest.TestCase):
    def setUp(self):
        self.adata = _make_adata()

    def test_recovery_all_cells(self):
        """Default source='imputed'."""
        df = fit_gene_betas(self.adata)
        g = df[df["gene"] == "G"].set_index("factor")
        for factor, val in KNOWN_BETA_G_IMPUTED.items():
            self.assertAlmostEqual(g.loc[factor, "beta"], val, places=6)
            self.assertAlmostEqual(g.loc[factor, "r2"], 1.0, places=6)
            self.assertEqual(g.loc[factor, "n_cells"], N)
            self.assertEqual(g.loc[factor, "group"], _group(factor))

        h = df[df["gene"] == "H"].set_index("factor")
        self.assertAlmostEqual(h.loc["TF_A", "beta"], KNOWN_BETA_H_IMPUTED["TF_A"], places=6)
        # metab@Glucose has a TRUE coefficient of 0 for H (y_H doesn't depend on it).
        self.assertAlmostEqual(h.loc[METAB_COL_IMPUTED, "beta"], 0.0, places=6)

    def test_recovery_all_cells_source_lognorm(self):
        """source='lognorm' must read the lognorm factor block AND the normalized_count
        layer -- both known betas AND the y values differ from the 'imputed' fixture, so
        recovering KNOWN_BETA_*_LOGNORM (not the imputed betas) proves both are wired
        consistently."""
        df = fit_gene_betas(self.adata, source="lognorm")
        g = df[df["gene"] == "G"].set_index("factor")
        for factor, val in KNOWN_BETA_G_LOGNORM.items():
            self.assertAlmostEqual(g.loc[factor, "beta"], val, places=6)
            self.assertAlmostEqual(g.loc[factor, "r2"], 1.0, places=6)
            self.assertEqual(g.loc[factor, "group"], _group(factor))
        # none of the imputed-source factor names should appear at all.
        self.assertFalse(set(FACTOR_COLS_IMPUTED) & set(g.index))

        h = df[df["gene"] == "H"].set_index("factor")
        self.assertAlmostEqual(h.loc["TF_A2", "beta"], KNOWN_BETA_H_LOGNORM["TF_A2"], places=6)
        self.assertAlmostEqual(h.loc[METAB_COL_LOGNORM, "beta"], 0.0, places=6)

    def test_unknown_source_raises_keyerror(self):
        """Change D: routed through build_x's `_source_layer` so the message is friendly
        ('unknown source ...'), not a bare KeyError('bogus')."""
        with self.assertRaises(KeyError) as ctx:
            fit_gene_betas(self.adata, genes=["G"], source="bogus")
        self.assertIn("unknown source", str(ctx.exception))

    def test_genes_none_defaults_to_this_sources_own_stored_genes(self):
        """MAJOR regression: 'Z' has an 'imputed' x_factor_map entry but NO 'lognorm'
        entry (as if build_factor_block(source='lognorm') found zero modulators for it).
        uns['x_genes'] deliberately does NOT include 'Z' (simulating a stale/otherwise-
        derived shared list) -- fit_gene_betas(source='imputed', genes=None) must still
        fit Z, because the default now comes from build_x.stored_genes(adata, 'imputed')
        (= that source's OWN x_factor_map keys), not any shared/other-source list."""
        n = 20
        rng = np.random.default_rng(77)
        tf = rng.normal(size=n)
        y_z = 2.0 * tf + 0.5  # exact linear -> r2 == 1, recoverable beta == 2.0

        adata = AnnData(X=np.zeros((n, 1), dtype=np.float64))
        adata.var_names = ["Z"]
        adata.obs_names = [f"c{i}" for i in range(n)]
        adata.layers["imputed_count"] = y_z.reshape(-1, 1)

        adata.obsm["x_factors"] = tf.reshape(-1, 1)
        adata.uns["x_factors_cols"] = ["TF_A"]
        adata.uns["x_factor_map"] = {"Z": ["TF_A"]}   # 'imputed' HAS Z
        adata.uns["x_factor_map_lognorm"] = {}         # 'lognorm' does NOT have Z
        adata.uns["x_genes"] = []                      # deliberately stale/empty shared list

        df = fit_gene_betas(adata, genes=None, metabolites=None, source="imputed")
        self.assertIn("Z", set(df["gene"]))
        z = df.set_index("factor")
        self.assertAlmostEqual(z.loc["TF_A", "beta"], 2.0, places=6)

    def test_metabolites_selection(self):
        only_glucose = fit_gene_betas(self.adata, genes=["G"], metabolites=["Glucose"])
        self.assertEqual(set(only_glucose["factor"]), set(FACTOR_COLS_IMPUTED) | {METAB_COL_IMPUTED})

        no_metab = fit_gene_betas(self.adata, genes=["G"], metabolites=None)
        self.assertEqual(set(no_metab["factor"]), set(FACTOR_COLS_IMPUTED))
        self.assertNotIn(METAB_COL_IMPUTED, set(no_metab["factor"]))

        all_metab = fit_gene_betas(self.adata, genes=["G"], metabolites="all")
        self.assertEqual(set(all_metab["factor"]), set(FACTOR_COLS_IMPUTED) | {METAB_COL_IMPUTED})

    def test_annotation_filter_restricts_cells_and_still_recovers(self):
        df = fit_gene_betas(self.adata, genes=["G"], annot_col="ct", annot_value="T")
        self.assertTrue((df["n_cells"] == N // 2).all())
        g = df.set_index("factor")
        for factor, val in KNOWN_BETA_G_IMPUTED.items():
            self.assertAlmostEqual(g.loc[factor, "beta"], val, places=5)

    def test_skips_gene_not_in_factor_map(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            df = fit_gene_betas(self.adata, genes=["NOPE"])
        self.assertTrue(df.empty)
        self.assertTrue(any("x_factor_map" in str(w.message) for w in caught))

    def test_skips_gene_with_too_few_cells(self):
        few_cells = self.adata.obs_names[:2]  # < n_factors(4)+1 for G
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            df = fit_gene_betas(self.adata, genes=["G"], cells=few_cells)
        self.assertTrue(df.empty)
        self.assertTrue(any("too few cells" in str(w.message) for w in caught))


class SubsampleCellsTests(unittest.TestCase):
    def setUp(self):
        self.adata = _make_adata()

    def test_reproducible_for_fixed_seed(self):
        a = subsample_cells(self.adata, seed=42, frac=0.5)
        b = subsample_cells(self.adata, seed=42, frac=0.5)
        np.testing.assert_array_equal(a, b)

    def test_frac_size(self):
        out = subsample_cells(self.adata, seed=0, frac=0.25)
        self.assertEqual(len(out), 10)  # round(0.25 * 40)

    def test_n_size(self):
        out = subsample_cells(self.adata, seed=0, n=7)
        self.assertEqual(len(out), 7)

    def test_respects_annotation_filter(self):
        out = subsample_cells(self.adata, seed=0, frac=1.0, annot_col="ct", annot_value="T")
        t_cells = set(self.adata.obs_names[self.adata.obs["ct"] == "T"])
        self.assertEqual(set(out), t_cells)

    def test_samples_without_replacement(self):
        out = subsample_cells(self.adata, seed=3, n=N)
        self.assertEqual(len(set(out)), N)

    def test_raises_on_empty_pool(self):
        with self.assertRaises(ValueError):
            subsample_cells(self.adata, seed=0, frac=0.5, annot_col="ct", annot_value="ZZZ")

    def test_raises_when_n_exceeds_pool(self):
        with self.assertRaises(ValueError):
            subsample_cells(self.adata, seed=0, n=N + 1)

    def test_raises_unless_exactly_one_of_frac_n(self):
        with self.assertRaises(ValueError):
            subsample_cells(self.adata, seed=0)
        with self.assertRaises(ValueError):
            subsample_cells(self.adata, seed=0, frac=0.5, n=5)


class FitStandardizeTests(unittest.TestCase):
    """`_fit(..., standardize=True)` z-scores X's columns before fitting, so the
    returned betas are per-SD; `standardize=False` (default) keeps raw-scale betas."""

    def setUp(self):
        self.adata = _make_adata()

    def test_raw_scale_recovers_known_betas(self):
        df = fit_gene_betas(self.adata, genes=["G"], standardize=False)
        raw = df.set_index("factor")["beta"]
        for factor, val in KNOWN_BETA_G_IMPUTED.items():
            self.assertAlmostEqual(raw.loc[factor], val, places=6)

    def test_standardized_betas_are_raw_betas_times_column_sd(self):
        raw = fit_gene_betas(self.adata, genes=["G"], standardize=False).set_index("factor")["beta"]
        std = fit_gene_betas(self.adata, genes=["G"], standardize=True).set_index("factor")["beta"]
        X = get_gene_factors(self.adata, "G", metabs="all")
        sd = X.to_numpy().std(axis=0)  # ddof=0, matching `_fit`'s np.std
        for i, factor in enumerate(X.columns):
            self.assertAlmostEqual(std.loc[factor], raw.loc[factor] * sd[i], places=4)
        # sanity: standardization actually changed at least one beta (columns aren't
        # already unit-SD), so this isn't a vacuous no-op check.
        self.assertFalse(np.allclose(raw.to_numpy(), std.to_numpy()))

    def test_fit_directly_standardize_matches_manual_zscore(self):
        rng = np.random.default_rng(3)
        X = rng.normal(loc=5.0, scale=2.0, size=(60, 3))
        beta_true = np.array([2.0, -1.0, 0.5])
        y = X @ beta_true + 1.0
        beta_raw, _ = _fit(X, y, method="OLS", standardize=False)
        beta_std, _ = _fit(X, y, method="OLS", standardize=True)
        np.testing.assert_allclose(beta_raw, beta_true, atol=1e-8)
        sd = X.std(axis=0)
        np.testing.assert_allclose(beta_std, beta_true * sd, atol=1e-6)


class FitL1SparsityTests(unittest.TestCase):
    """`method='l1'` (Lasso) with a moderate penalty zeroes irrelevant factors while
    keeping relevant ones nonzero; increasing the penalty shrinks magnitudes further."""

    def setUp(self):
        rng = np.random.default_rng(11)
        n, k = 400, 5
        self.X = rng.normal(size=(n, k))
        # only columns 0 and 3 are relevant.
        self.beta_true = np.array([3.0, 0.0, 0.0, -2.0, 0.0])
        self.y = self.X @ self.beta_true

    def test_moderate_penalty_zeroes_irrelevant_keeps_relevant(self):
        beta, _ = _fit(self.X, self.y, method="l1", penalty=0.5, standardize=True)
        for i in (1, 2, 4):
            self.assertEqual(beta[i], 0.0)
        for i in (0, 3):
            self.assertNotEqual(beta[i], 0.0)
        # signs preserved on the surviving coefficients.
        self.assertGreater(beta[0], 0)
        self.assertLess(beta[3], 0)

    def test_larger_penalty_shrinks_magnitude(self):
        beta_small, _ = _fit(self.X, self.y, method="l1", penalty=0.2, standardize=True)
        beta_large, _ = _fit(self.X, self.y, method="l1", penalty=2.0, standardize=True)
        self.assertLess(abs(beta_large[0]), abs(beta_small[0]))
        self.assertLess(abs(beta_large[3]), abs(beta_small[3]))


class FitMethodErrorTests(unittest.TestCase):
    def test_unknown_method_raises_value_error(self):
        X = np.zeros((5, 2))
        y = np.zeros(5)
        with self.assertRaises(ValueError):
            _fit(X, y, method="not_a_method")

    def test_fit_gene_betas_propagates_value_error(self):
        adata = _make_adata()
        with self.assertRaises(ValueError):
            fit_gene_betas(adata, genes=["G"], method="not_a_method")


class SubsampleGeneBetasTests(unittest.TestCase):
    def setUp(self):
        self.adata = _make_adata()

    def test_shape_and_default_includes_all_factor_groups(self):
        df = subsample_gene_betas(self.adata, "G", n_subsamples=5, frac=0.8, seed0=0)
        self.assertEqual(df.shape, (5, 4))
        groups = {_group(c) for c in df.columns}
        self.assertIn("tf", groups)
        self.assertIn("lr", groups)
        self.assertIn("metab", groups)

    def test_source_lognorm_recovers_known_betas(self):
        """No noise + frac=1.0 -> every subsample refits on all N cells, so betas should
        match KNOWN_BETA_G_LOGNORM (not the 'imputed' fixture's betas) to floating
        precision."""
        df = subsample_gene_betas(self.adata, "G", source="lognorm",
                                  n_subsamples=3, frac=1.0, seed0=0)
        groups = {_group(c) for c in df.columns}
        self.assertEqual(groups, {"tf", "lr", "metab"})
        for factor, val in KNOWN_BETA_G_LOGNORM.items():
            self.assertTrue(np.allclose(df[factor].to_numpy(), val, atol=1e-6))

    def test_factor_subset_returns_just_those_columns_in_order(self):
        df = subsample_gene_betas(self.adata, "G", factors=["L$R", "TF_A"],
                                  n_subsamples=3, frac=0.8, seed0=0)
        self.assertEqual(list(df.columns), ["L$R", "TF_A"])

    def test_unknown_source_raises_friendly_keyerror(self):
        """Change D: routed through build_x's `_source_layer`, so the message is
        friendly ('unknown source ...'), not a bare KeyError('bogus')."""
        with self.assertRaises(KeyError) as ctx:
            subsample_gene_betas(self.adata, "G", source="bogus", n_subsamples=1)
        self.assertIn("unknown source", str(ctx.exception))

    def test_bad_factor_name_warns_and_is_dropped(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            df = subsample_gene_betas(self.adata, "G", factors=["TF_A", "NOPE"],
                                      n_subsamples=3, frac=0.8, seed0=0)
        self.assertEqual(list(df.columns), ["TF_A"])
        self.assertTrue(any("NOPE" in str(w.message) for w in caught))

    def test_method_penalty_standardize_honored(self):
        df = subsample_gene_betas(self.adata, "G", n_subsamples=5, frac=1.0, seed0=0,
                                  method="l1", penalty=1000.0, standardize=True)
        self.assertGreater((df.abs() < 1e-8).to_numpy().mean(), 0.5)

    def test_too_small_draw_yields_nan_rows(self):
        # frac=0.05 of 40 cells -> 2 cells, < n_factors(4)+1 for G.
        df = subsample_gene_betas(self.adata, "G", n_subsamples=3, frac=0.05, seed0=0)
        self.assertTrue(np.all(np.isnan(df.to_numpy())))

    def test_deterministic_given_seed0(self):
        a = subsample_gene_betas(self.adata, "G", n_subsamples=5, frac=0.8, seed0=0)
        b = subsample_gene_betas(self.adata, "G", n_subsamples=5, frac=0.8, seed0=0)
        pd.testing.assert_frame_equal(a, b)


class SelectCellsGuardTests(unittest.TestCase):
    def setUp(self):
        self.adata = _make_adata()

    def test_annot_value_none_raises(self):
        with self.assertRaises(ValueError):
            _select_cells(self.adata, "ct", None, None)

    def test_wrong_length_boolean_mask_raises(self):
        bad_mask = np.array([True, False, True])  # len 3 != n_obs (40)
        with self.assertRaises(ValueError):
            _select_cells(self.adata, None, None, bad_mask)

    def test_unknown_obs_names_warn_and_are_dropped(self):
        real = list(self.adata.obs_names[:5])
        cells = real + ["not_a_real_cell"]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = _select_cells(self.adata, None, None, cells)
        self.assertEqual(set(out), set(real))
        self.assertTrue(any("not found" in str(w.message) for w in caught))


class RankCoefficientsTests(unittest.TestCase):
    def test_sorted_by_abs_beta_descending(self):
        df = pd.DataFrame({
            "gene": ["G", "G", "G"],
            "factor": ["a", "b", "c"],
            "group": ["tf", "tf", "tf"],
            "beta": [0.5, -3.0, 1.2],
            "r2": [1.0, 1.0, 1.0],
            "n_cells": [10, 10, 10],
        })
        ranked = rank_coefficients(df)
        self.assertEqual(list(ranked["factor"]), ["b", "c", "a"])
        # original untouched (a copy)
        self.assertEqual(list(df["factor"]), ["a", "b", "c"])


class PlotSmokeTests(unittest.TestCase):
    def setUp(self):
        import matplotlib
        matplotlib.use("Agg")

    def test_plot_top_coefficients_returns_ax(self):
        df = pd.DataFrame({
            "gene": ["G"] * 3,
            "factor": ["a", "b", "c"],
            "group": ["tf", "tf", "metab"],
            "beta": [0.5, -3.0, 1.2],
            "r2": [1.0] * 3,
            "n_cells": [10] * 3,
        })
        ax = plot_top_coefficients(df, top=2)
        self.assertIsNotNone(ax)

    def test_plot_beta_histogram_returns_ax_and_drops_nan(self):
        values = np.array([1.0, 2.0, np.nan, 3.0])
        ax = plot_beta_histogram(values, title="test")
        self.assertIsNotNone(ax)


if __name__ == "__main__":
    unittest.main()
