"""Tier-1 tests for `metab_processing/LinearRegression/least_squares.py`.

Builds a synthetic AnnData directly against the `x_factors`/`x_metab` store contract
that `build_x.py` produces (no estimator/torch involved -- `get_gene_factors` is pure
pandas), with a KNOWN linear generative model so OLS must recover the true betas.

Fixture: 40 cells, an obs['ct'] annotation ('T'/'B', 20 each), two genes:
  - 'G' uses factors ['TF_A', 'TF_B', 'L$R'] (x_factor_map['G']) plus the shared metab
    column 'metab@Glucose'; y_G is an EXACT linear combination of all four (known_beta_G)
    plus an intercept -- no noise, so OLS recovery is exact to floating precision.
  - 'H' uses only ['TF_A']; y_H depends only on TF_A (known_beta_H['TF_A']), so when
    'all' metabolites are requested the extra metab@Glucose column has a true
    coefficient of 0 (tests OLS handling an unused predictor).
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
from metab_processing.LinearRegression.least_squares import (
    fit_gene_betas,
    _fit_ols,
    _select_cells,
    subsample_cells,
    subsample_metab_betas,
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

    y_g = (factors @ np.array([KNOWN_BETA_G["TF_A"], KNOWN_BETA_G["TF_B"], KNOWN_BETA_G["L$R"]])
           + metab[:, 0] * KNOWN_BETA_G[METAB_COL] + INTERCEPT_G)
    y_h = factors[:, 0] * KNOWN_BETA_H["TF_A"] + INTERCEPT_H

    obs_names = [f"c{i}" for i in range(N)]
    adata = AnnData(X=np.zeros((N, 2), dtype=np.float64))
    adata.var_names = ["G", "H"]
    adata.obs_names = obs_names
    adata.obs["ct"] = ["T"] * (N // 2) + ["B"] * (N // 2)
    adata.layers["imputed_count"] = np.column_stack([y_g, y_h])

    adata.obsm["x_factors"] = factors
    adata.uns["x_factors_cols"] = list(FACTOR_COLS)
    adata.uns["x_factor_map"] = {"G": list(FACTOR_COLS), "H": ["TF_A"]}
    adata.uns["x_genes"] = ["G", "H"]

    adata.obsm["x_metab"] = metab
    adata.uns["x_metab_modulators"] = [METAB_COL]

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
            self.assertAlmostEqual(g.loc[factor, "r2"], 1.0, places=6)
            self.assertEqual(g.loc[factor, "n_cells"], N)
            self.assertEqual(g.loc[factor, "group"], _group(factor))

        h = df[df["gene"] == "H"].set_index("factor")
        self.assertAlmostEqual(h.loc["TF_A", "beta"], KNOWN_BETA_H["TF_A"], places=6)
        # metab@Glucose has a TRUE coefficient of 0 for H (y_H doesn't depend on it).
        self.assertAlmostEqual(h.loc[METAB_COL, "beta"], 0.0, places=6)

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


class SubsampleMetabBetasTests(unittest.TestCase):
    def setUp(self):
        self.adata = _make_adata()

    def test_length_and_deterministic(self):
        a = subsample_metab_betas(self.adata, "G", "Glucose", n_subsamples=5, frac=0.8, seed0=0)
        b = subsample_metab_betas(self.adata, "G", "Glucose", n_subsamples=5, frac=0.8, seed0=0)
        self.assertEqual(len(a), 5)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.any(np.isnan(a)))

    def test_nan_when_metabolite_column_absent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = subsample_metab_betas(
                self.adata, "G", "Iron", n_subsamples=3, frac=0.8, seed0=0)
        self.assertEqual(len(out), 3)
        self.assertTrue(np.all(np.isnan(out)))


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
