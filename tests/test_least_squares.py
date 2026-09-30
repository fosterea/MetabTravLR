"""Tier-1 tests for `metab_processing/LinearRegression/least_squares.py`.

Builds a synthetic AnnData directly against the store contract that `build_x.py` produces
(no estimator/torch involved -- `get_gene_factors` is pure pandas), with a KNOWN linear
generative model so OLS must recover the true betas.

Rev 6 (2026-09-29): dropped the two-SOURCE ('imputed'/'lognorm') dual-block machinery.
There is now ONE factor block (the plain unsuffixed keys: `x_factors`, `x_factors_cols`,
`x_factor_map`, `x_metab`, `x_metab_modulators`, `x_genes`) and y is read from the single
`normalized_count` layer (`build_x.LAYER`).

Fixture: 40 cells, an obs['ct'] annotation ('T'/'B', 20 each), two genes:
  - 'G' uses all 3 factor columns plus the shared metab column; y_G is an EXACT linear
    combination of all four (known betas) plus an intercept -- no noise, so OLS recovery
    is exact to floating precision.
  - 'H' uses only the first TF factor column; y_H depends only on that column, so when
    'all' metabolites are requested the extra metab column has a true coefficient of 0
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
    resample_cells,
    subsample_gene_betas,
    rank_coefficients,
    plot_top_coefficients,
    plot_beta_histogram,
)

N = 40

FACTOR_COLS = ["TF_A", "TF_B", "L$R"]
METAB_COL = "metab@Glucose"
KNOWN_BETA_G = {"TF_A": 1.5, "TF_B": -2.0, "L$R": 0.7, METAB_COL: 0.9}
INTERCEPT_G = 0.3
KNOWN_BETA_H = {"TF_A": -1.2}
INTERCEPT_H = 0.05


def _make_adata():
    rng = np.random.default_rng(0)
    factors = rng.normal(size=(N, 3))   # columns: TF_A, TF_B, L$R
    metab = rng.normal(size=(N, 1))     # column: metab@Glucose

    y_g = (factors @ np.array([KNOWN_BETA_G[c] for c in FACTOR_COLS])
           + metab[:, 0] * KNOWN_BETA_G[METAB_COL] + INTERCEPT_G)
    y_h = factors[:, 0] * KNOWN_BETA_H["TF_A"] + INTERCEPT_H

    obs_names = [f"c{i}" for i in range(N)]
    adata = AnnData(X=np.zeros((N, 2), dtype=np.float64))
    adata.var_names = ["G", "H"]
    adata.obs_names = obs_names
    adata.obs["ct"] = ["T"] * (N // 2) + ["B"] * (N // 2)
    # y is read from LAYER ('normalized_count'); the fixture carries the known-beta y there.
    adata.layers["normalized_count"] = np.column_stack([y_g, y_h])

    adata.obsm["x_factors"] = factors
    adata.uns["x_factors_cols"] = list(FACTOR_COLS)
    adata.uns["x_factor_map"] = {"G": list(FACTOR_COLS), "H": ["TF_A"]}
    adata.obsm["x_metab"] = metab
    adata.uns["x_metab_modulators"] = [METAB_COL]
    adata.uns["x_genes"] = ["G", "H"]

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
        df = fit_gene_betas(self.adata)
        g = df[df["gene"] == "G"].set_index("factor")
        for factor, val in KNOWN_BETA_G.items():
            self.assertAlmostEqual(g.loc[factor, "beta"], val, places=6)
            self.assertAlmostEqual(g.loc[factor, "model_r2"], 1.0, places=6)
            # per-factor Pearson: r2 == r**2, both in [0, 1]
            self.assertAlmostEqual(g.loc[factor, "r2"], g.loc[factor, "r"] ** 2, places=10)
            self.assertTrue(0.0 <= g.loc[factor, "r2"] <= 1.0 + 1e-9)
            self.assertEqual(g.loc[factor, "n_cells"], N)
            self.assertEqual(g.loc[factor, "group"], _group(factor))

        h = df[df["gene"] == "H"].set_index("factor")
        self.assertAlmostEqual(h.loc["TF_A", "beta"], KNOWN_BETA_H["TF_A"], places=6)
        # metab@Glucose has a TRUE coefficient of 0 for H (y_H doesn't depend on it).
        self.assertAlmostEqual(h.loc[METAB_COL, "beta"], 0.0, places=6)

    def test_genes_none_defaults_to_stored_genes(self):
        """'Z' has an x_factor_map entry but is deliberately absent from uns['x_genes']
        (simulating a stale/otherwise-derived list) -- fit_gene_betas(genes=None) must
        still fit Z, because the default now comes from build_x.stored_genes(adata)
        (= x_factor_map's own keys), not any separately-tracked gene list."""
        n = 20
        rng = np.random.default_rng(77)
        tf = rng.normal(size=n)
        y_z = 2.0 * tf + 0.5  # exact linear -> r2 == 1, recoverable beta == 2.0

        adata = AnnData(X=np.zeros((n, 1), dtype=np.float64))
        adata.var_names = ["Z"]
        adata.obs_names = [f"c{i}" for i in range(n)]
        adata.layers["normalized_count"] = y_z.reshape(-1, 1)

        adata.obsm["x_factors"] = tf.reshape(-1, 1)
        adata.uns["x_factors_cols"] = ["TF_A"]
        adata.uns["x_factor_map"] = {"Z": ["TF_A"]}
        adata.uns["x_genes"] = []  # deliberately stale/empty

        df = fit_gene_betas(adata, genes=None, metabolites=None)
        self.assertIn("Z", set(df["gene"]))
        z = df.set_index("factor")
        self.assertAlmostEqual(z.loc["TF_A", "beta"], 2.0, places=6)

    def test_metabolites_selection(self):
        only_glucose = fit_gene_betas(self.adata, genes=["G"], metabolites=["Glucose"])
        self.assertEqual(set(only_glucose["factor"]), set(FACTOR_COLS) | {METAB_COL})

        no_metab = fit_gene_betas(self.adata, genes=["G"], metabolites=None)
        self.assertEqual(set(no_metab["factor"]), set(FACTOR_COLS))
        self.assertNotIn(METAB_COL, set(no_metab["factor"]))

        all_metab = fit_gene_betas(self.adata, genes=["G"], metabolites="all")
        self.assertEqual(set(all_metab["factor"]), set(FACTOR_COLS) | {METAB_COL})

    def test_annotation_filter_restricts_cells_and_still_recovers(self):
        df = fit_gene_betas(self.adata, genes=["G"], annot_col="ct", annot_value="T")
        self.assertTrue((df["n_cells"] == N // 2).all())
        g = df.set_index("factor")
        for factor, val in KNOWN_BETA_G.items():
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


class ResampleCellsTests(unittest.TestCase):
    def setUp(self):
        self.adata = _make_adata()

    def test_size_equals_pool_size(self):
        out = resample_cells(self.adata, seed=0)
        self.assertEqual(len(out), N)

    def test_reproducible_for_fixed_seed(self):
        a = resample_cells(self.adata, seed=42)
        b = resample_cells(self.adata, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_all_returned_names_in_pool(self):
        pool = set(self.adata.obs_names)
        out = resample_cells(self.adata, seed=1)
        self.assertTrue(set(out) <= pool)

    def test_contains_duplicates(self):
        # with N=40 draws with replacement from a pool of 40, some duplicate is
        # overwhelmingly likely; check across a few seeds to avoid flakiness.
        found_dup = any(
            len(set(resample_cells(self.adata, seed=s))) < N for s in range(5))
        self.assertTrue(found_dup)

    def test_respects_annotation_filter(self):
        out = resample_cells(self.adata, seed=0, annot_col="ct", annot_value="T")
        t_cells = set(self.adata.obs_names[self.adata.obs["ct"] == "T"])
        self.assertTrue(set(out) <= t_cells)
        self.assertEqual(len(out), len(t_cells))

    def test_raises_on_empty_pool(self):
        with self.assertRaises(ValueError):
            resample_cells(self.adata, seed=0, annot_col="ct", annot_value="ZZZ")


class FitStandardizeTests(unittest.TestCase):
    """`_fit(..., standardize=True)` z-scores X's columns before fitting, so the
    returned betas are per-SD; `standardize=False` (default) keeps raw-scale betas."""

    def setUp(self):
        self.adata = _make_adata()

    def test_raw_scale_recovers_known_betas(self):
        df = fit_gene_betas(self.adata, genes=["G"], standardize=False)
        raw = df.set_index("factor")["beta"]
        for factor, val in KNOWN_BETA_G.items():
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
        df = subsample_gene_betas(self.adata, "G", n_subsamples=5, seed0=0)
        self.assertEqual(df.shape, (5, 4))
        groups = {_group(c) for c in df.columns}
        self.assertIn("tf", groups)
        self.assertIn("lr", groups)
        self.assertIn("metab", groups)

    def test_factor_subset_returns_just_those_columns_in_order(self):
        df = subsample_gene_betas(self.adata, "G", factors=["L$R", "TF_A"],
                                  n_subsamples=3, seed0=0)
        self.assertEqual(list(df.columns), ["L$R", "TF_A"])

    def test_bad_factor_name_warns_and_is_dropped(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            df = subsample_gene_betas(self.adata, "G", factors=["TF_A", "NOPE"],
                                      n_subsamples=3, seed0=0)
        self.assertEqual(list(df.columns), ["TF_A"])
        self.assertTrue(any("NOPE" in str(w.message) for w in caught))

    def test_method_penalty_standardize_honored(self):
        df = subsample_gene_betas(self.adata, "G", n_subsamples=5, seed0=0,
                                  method="l1", penalty=1000.0, standardize=True)
        self.assertGreater((df.abs() < 1e-8).to_numpy().mean(), 0.5)

    def test_too_few_cells_in_pool_yields_nan_rows(self):
        """A resample is always the same size as the eligible pool, so the only way to
        get a too-small draw is a too-small pool -- restrict via annot_col/annot_value to
        a 2-cell pool, < n_factors(4)+1 for G."""
        adata = self.adata.copy()
        adata.obs["ct2"] = ["small"] * 2 + ["rest"] * (N - 2)
        df = subsample_gene_betas(adata, "G", n_subsamples=3, seed0=0,
                                  annot_col="ct2", annot_value="small")
        self.assertTrue(np.all(np.isnan(df.to_numpy())))

    def test_deterministic_given_seed0(self):
        a = subsample_gene_betas(self.adata, "G", n_subsamples=5, seed0=0)
        b = subsample_gene_betas(self.adata, "G", n_subsamples=5, seed0=0)
        pd.testing.assert_frame_equal(a, b)

    def test_betas_vary_across_resamples(self):
        """With a resample-with-replacement (not the exact all-cells fit), per-factor
        betas should vary from row to row across a reasonably large n_subsamples --
        confirms subsample_gene_betas is actually driven by resample_cells rather than
        e.g. accidentally refitting on the same (full) cell set every time. `_make_adata`'s
        'G' is a NOISE-FREE exact linear function of its factors, so an OLS fit on any
        full-rank subset (duplicates included) recovers the exact true beta every time --
        that would make this test vacuous. So we add a little noise to y_G here, just for
        this test, to make the fit sensitive to which rows (and how many duplicates of
        each) a given resample happens to draw."""
        adata = self.adata.copy()
        rng = np.random.default_rng(99)
        noise = rng.normal(scale=0.5, size=N)
        y = pd.DataFrame(adata.layers["normalized_count"], columns=adata.var_names)
        y["G"] = y["G"] + noise
        adata.layers["normalized_count"] = y.to_numpy()

        df = subsample_gene_betas(adata, "G", n_subsamples=25, seed0=0)
        stds = df.std(axis=0)
        self.assertTrue((stds > 1e-9).all())


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
