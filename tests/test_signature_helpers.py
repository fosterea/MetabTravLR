"""Tests for `metab_processing/LinearRegression/signature_helpers.py` -- the shared
analysis layer for the signature notebooks. Synthetic data only (no MSigDB / cluster data);
the MSigDB-fetch path (`load_msigdb`) is network and not exercised here."""
import os
import sys
import unittest
import warnings

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import anndata as ad

from metab_processing.LinearRegression import signature_helpers as sh
from metab_processing.LinearRegression.build_x import LAYER

N = 80
SIG_GENES = ["A", "B", "C", "D"]           # the target signature's genes
OTHER = ["TF1", "TF2", "LIG", "REC"]       # genes referenced by factors
GENES = SIG_GENES + OTHER


def _make_store_adata(seed=0):
    """Tiny adata with a hand-built factor store + spatial + a cell_type with two T subtypes
    and an 'other' type, so run_signature's pooled / covariate / per-type runs all have cells."""
    rng = np.random.default_rng(seed)
    X = (rng.random((N, len(GENES))).astype(np.float32) + 0.1)
    a = ad.AnnData(X=X.copy())
    a.var_names = GENES
    a.obs_names = [f"c{i}" for i in range(N)]
    a.layers["normalized_count"] = X.copy()
    a.obsm["spatial"] = rng.uniform(0, 300, size=(N, 2))
    cts = np.array(["T1", "T2", "other"])[rng.integers(0, 3, size=N)]
    a.obs["cell_type"] = pd.Categorical(cts)

    # Factor store: two bare-TF factor columns. A/B reference TF1; note A's factor list also
    # includes the bare gene 'B' (a factor that references gene B) so the factor-split test
    # can see B removed.
    cols = ["TF1", "TF2", "B"]
    a.obsm["x_factors"] = X[:, [GENES.index("TF1"), GENES.index("TF2"), GENES.index("B")]].copy()
    a.uns["x_factors_cols"] = cols
    a.uns["x_factor_map"] = {"A": ["TF1", "B"], "B": ["TF1"], "C": ["TF2"], "D": ["TF2"]}
    a.uns["x_genes"] = ["A", "B", "C", "D"]
    return a


class TableAndFilterTests(unittest.TestCase):
    def test_add_subsample_percentiles_columns_and_values(self):
        betas = pd.DataFrame({"factor": ["TF1", "TF2"], "beta": [1.0, -2.0]})
        sub = pd.DataFrame({"TF1": [0.9, 1.1, 1.0], "TF2": [-2.1, -1.9, -2.0]})
        sub_r = pd.DataFrame({"TF1": [0.5, 0.6, 0.55], "TF2": [-0.4, -0.3, -0.35]})
        out = sh.add_subsample_percentiles(betas, sub, sub_r, lower=15, upper=85)
        for c in ("min_subsample_beta", "max_subsample_beta", "avg_subsample_beta",
                  "var_subsample_beta", "msd_subsample_beta", "avg_subsample_r"):
            self.assertIn(c, out.columns)
        self.assertAlmostEqual(out.loc[out.factor == "TF1", "avg_subsample_beta"].iloc[0], 1.0, places=6)
        # msd about point beta 1.0 == mean((sub-1)^2)
        self.assertAlmostEqual(out.loc[out.factor == "TF1", "msd_subsample_beta"].iloc[0],
                               float(np.mean((np.array([0.9, 1.1, 1.0]) - 1.0) ** 2)), places=6)

    def test_meansq_subsample_beta_value_and_identity(self):
        betas = pd.DataFrame({"factor": ["F"], "beta": [1.0]})
        sub = pd.DataFrame({"F": [0.0, 2.0, 4.0]})
        out = sh.add_subsample_percentiles(betas, sub)
        got = out["meansq_subsample_beta"].iloc[0]
        self.assertAlmostEqual(got, float(np.mean(np.square([0.0, 2.0, 4.0]))), places=6)
        # mean(beta^2) == var + mean^2
        self.assertAlmostEqual(got, float(np.var([0.0, 2.0, 4.0]) + np.mean([0.0, 2.0, 4.0]) ** 2),
                               places=6)

    def test_filter_nonzero_interval_cross_vs_touch(self):
        res = pd.DataFrame({
            "factor": ["cross", "touch", "pos"],
            "min_subsample_beta": [-1.0, 0.0, 0.5],
            "max_subsample_beta": [1.0, 2.0, 3.0],
        })
        cross = sh.filter_nonzero_interval(res, touch_zero=False)
        self.assertEqual(set(cross.factor), {"touch", "pos"})   # touching kept
        touch = sh.filter_nonzero_interval(res, touch_zero=True)
        self.assertEqual(set(touch.factor), {"pos"})            # touching dropped


class SubsampleStatsTests(unittest.TestCase):
    """`subsample_stats` computes known-answer per-factor stats from a raw `n_sub x
    factor` betas frame: `ci_lo`/`ci_hi` (percentiles), `selection_freq` (mean(b != 0)),
    `p_stability` (2*min(mean(b<=0), mean(b>=0)), clipped to [0,1])."""

    def test_known_values_positive_factor(self):
        # 8 values, all > 0 -> selection_freq=1, p_stability = 2*min(0, 1) clipped = 0.
        vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        betas = pd.DataFrame({"POS": vals})
        out = sh.subsample_stats(betas, ci=(0, 100)).set_index("factor")
        self.assertAlmostEqual(out.loc["POS", "ci_lo"], np.nanpercentile(vals, 0), places=8)
        self.assertAlmostEqual(out.loc["POS", "ci_hi"], np.nanpercentile(vals, 100), places=8)
        self.assertAlmostEqual(out.loc["POS", "selection_freq"], 1.0, places=8)
        self.assertAlmostEqual(out.loc["POS", "p_stability"], 0.0, places=8)

    def test_known_values_mixed_sign_factor(self):
        # half negative, half positive -> p_stability = 2*min(0.5,0.5) = 1.0 (max instability).
        vals = [-2.0, -1.0, 1.0, 2.0]
        betas = pd.DataFrame({"MIXED": vals})
        out = sh.subsample_stats(betas).set_index("factor")
        self.assertAlmostEqual(out.loc["MIXED", "selection_freq"], 1.0, places=8)
        self.assertAlmostEqual(out.loc["MIXED", "p_stability"], 1.0, places=8)

    def test_all_zero_factor(self):
        betas = pd.DataFrame({"ZERO": [0.0, 0.0, 0.0, 0.0]})
        out = sh.subsample_stats(betas).set_index("factor")
        self.assertAlmostEqual(out.loc["ZERO", "selection_freq"], 0.0, places=8)
        # all values <= 0 AND all values >= 0 -> 2*min(1,1) clipped to 1 -> max "stability".
        self.assertAlmostEqual(out.loc["ZERO", "p_stability"], 1.0, places=8)
        self.assertAlmostEqual(out.loc["ZERO", "ci_lo"], 0.0, places=8)
        self.assertAlmostEqual(out.loc["ZERO", "ci_hi"], 0.0, places=8)

    def test_nan_rows_excluded_from_stats(self):
        betas = pd.DataFrame({"F": [1.0, np.nan, 1.0, np.nan]})
        out = sh.subsample_stats(betas).set_index("factor")
        self.assertAlmostEqual(out.loc["F", "selection_freq"], 1.0, places=8)

    def test_multiple_factors_independent_columns(self):
        betas = pd.DataFrame({"POS": [1.0, 2.0, 3.0], "ZERO": [0.0, 0.0, 0.0]})
        out = sh.subsample_stats(betas).set_index("factor")
        self.assertEqual(set(out.index), {"POS", "ZERO"})
        self.assertAlmostEqual(out.loc["POS", "selection_freq"], 1.0, places=8)
        self.assertAlmostEqual(out.loc["ZERO", "selection_freq"], 0.0, places=8)


class SweepFilterTests(unittest.TestCase):
    """`sweep_filter` summarizes a `sweep_gene_betas`-shaped frame (columns `factor`,
    `beta`, `penalty`) per factor: min/max beta, sign-flip count, frac_zero, stable_sign."""

    def _sweep_df(self, factor, penalties, betas):
        return pd.DataFrame({"factor": [factor] * len(betas), "penalty": penalties,
                             "beta": betas})

    def test_sign_flip_detected(self):
        df = self._sweep_df("FLIP", [0.1, 0.5, 1.0, 2.0], [2.0, 1.0, -1.0, -2.0])
        out = sh.sweep_filter(df).set_index("factor")
        self.assertEqual(out.loc["FLIP", "n_sign_flips"], 1)
        self.assertFalse(out.loc["FLIP", "stable_sign"])
        self.assertAlmostEqual(out.loc["FLIP", "min_beta"], -2.0, places=8)
        self.assertAlmostEqual(out.loc["FLIP", "max_beta"], 2.0, places=8)
        self.assertAlmostEqual(out.loc["FLIP", "frac_zero"], 0.0, places=8)

    def test_zeroing_factor_no_flip(self):
        df = self._sweep_df("ZEROING", [0.1, 0.5, 1.0, 2.0], [3.0, 2.0, 0.0, 0.0])
        out = sh.sweep_filter(df).set_index("factor")
        self.assertEqual(out.loc["ZEROING", "n_sign_flips"], 0)
        self.assertTrue(out.loc["ZEROING", "stable_sign"])
        self.assertAlmostEqual(out.loc["ZEROING", "frac_zero"], 0.5, places=8)

    def test_zero_between_opposite_signs_counts_one_flip(self):
        # +,0,- : the zero is a gap (not a third state) -> 1 flip, frac_zero reports it.
        df = self._sweep_df("PASS0", [0.1, 0.5, 1.0], [1.0, 0.0, -1.0])
        out = sh.sweep_filter(df).set_index("factor")
        self.assertEqual(out.loc["PASS0", "n_sign_flips"], 1)
        self.assertAlmostEqual(out.loc["PASS0", "frac_zero"], 1.0 / 3.0, places=8)
        self.assertFalse(out.loc["PASS0", "stable_sign"])

    def test_nan_betas_are_dropped_no_phantom_flip(self):
        df = self._sweep_df("NANS", [0.1, 0.5, 1.0], [1.0, np.nan, -1.0])
        out = sh.sweep_filter(df).set_index("factor")
        self.assertEqual(out.loc["NANS", "n_sign_flips"], 1)        # NaN dropped, not a phantom flip
        self.assertAlmostEqual(out.loc["NANS", "min_beta"], -1.0, places=8)
        self.assertAlmostEqual(out.loc["NANS", "max_beta"], 1.0, places=8)

    def test_all_nan_factor(self):
        df = self._sweep_df("ALLNAN", [0.1, 1.0], [np.nan, np.nan])
        out = sh.sweep_filter(df).set_index("factor")
        self.assertEqual(out.loc["ALLNAN", "n_sign_flips"], 0)
        self.assertTrue(out.loc["ALLNAN", "stable_sign"])
        self.assertTrue(np.isnan(out.loc["ALLNAN", "min_beta"]))

    def test_stable_positive_factor(self):
        df = self._sweep_df("STABLE", [0.1, 1.0, 10.0], [3.0, 2.0, 1.0])
        out = sh.sweep_filter(df).set_index("factor")
        self.assertTrue(out.loc["STABLE", "stable_sign"])
        self.assertEqual(out.loc["STABLE", "n_sign_flips"], 0)
        self.assertAlmostEqual(out.loc["STABLE", "frac_zero"], 0.0, places=8)

    def test_tol_treats_small_values_as_zero(self):
        df = self._sweep_df("TOL", [0.1, 1.0], [0.001, -0.001])
        out_no_tol = sh.sweep_filter(df, tol=0.0).set_index("factor")
        out_tol = sh.sweep_filter(df, tol=0.01).set_index("factor")
        self.assertEqual(out_no_tol.loc["TOL", "n_sign_flips"], 1)
        self.assertEqual(out_tol.loc["TOL", "n_sign_flips"], 0)
        self.assertAlmostEqual(out_tol.loc["TOL", "frac_zero"], 1.0, places=8)

    def test_multiple_factors_grouped_independently(self):
        flip = self._sweep_df("A", [0.1, 1.0], [1.0, -1.0])
        zeroing = self._sweep_df("B", [0.1, 1.0], [1.0, 0.0])
        df = pd.concat([flip, zeroing], ignore_index=True)
        out = sh.sweep_filter(df).set_index("factor")
        self.assertEqual(set(out.index), {"A", "B"})
        self.assertEqual(out.loc["A", "n_sign_flips"], 1)
        self.assertEqual(out.loc["B", "n_sign_flips"], 0)

    def test_unordered_penalty_input_is_sorted_before_flip_counting(self):
        """Rows given out of penalty order must be sorted internally -- same result as if
        they'd been passed in ascending-penalty order."""
        shuffled = pd.DataFrame({"factor": ["A"] * 3, "penalty": [1.0, 0.1, 0.5],
                                 "beta": [-1.0, 2.0, 1.0]})
        ordered = pd.DataFrame({"factor": ["A"] * 3, "penalty": [0.1, 0.5, 1.0],
                                "beta": [2.0, 1.0, -1.0]})
        out_shuffled = sh.sweep_filter(shuffled).set_index("factor")
        out_ordered = sh.sweep_filter(ordered).set_index("factor")
        self.assertEqual(out_shuffled.loc["A", "n_sign_flips"], out_ordered.loc["A", "n_sign_flips"])
        self.assertEqual(out_shuffled.loc["A", "n_sign_flips"], 1)


class FactorSplitTests(unittest.TestCase):
    def test_signature_gene_referenced_by_factor_is_removed(self):
        adata = _make_store_adata()
        removed, remaining = sh.signature_factor_split(adata, ["A", "B"])
        self.assertEqual(removed, ["B"])       # B is referenced by A's 'B' factor column
        self.assertEqual(remaining, ["A"])

    def test_report_factor_removal_runs(self):
        adata = _make_store_adata()
        sh.report_factor_removal(adata, {"SIG": ["A", "B"]})   # just must not raise


class AddCancerFactorsTests(unittest.TestCase):
    def test_adds_celltype_split_and_proximity_custom_columns(self):
        adata = _make_store_adata()
        sh.add_cancer_factors(adata, {"CANCER": ["C", "D"]}, annot_col="cell_type",
                              add_proximity=True, verbose=False)
        cols = list(adata.uns["x_custom_cols"])
        # one {ct}-CANCER column per present cell type, plus one {ct}-proximity each.
        for ct in ("T1", "T2", "other"):
            self.assertIn(f"{ct}-CANCER", cols)
            self.assertIn(f"{ct}-proximity", cols)
        self.assertEqual(adata.obsm["x_custom"].shape[0], N)


class RunSignatureTests(unittest.TestCase):
    def test_multi_run_returns_expected_runs_and_includes_custom_factors(self):
        adata = _make_store_adata(1)
        sh.add_cancer_factors(adata, {"CANCER": ["C", "D"]}, annot_col="cell_type",
                              add_proximity=True, verbose=False)
        sets = {"TARGET": ["A", "B", "C"]}
        results = sh.run_signature(
            adata, "TARGET", sets, cell_types=["T1", "T2"], cell_type_col="cell_type",
            method="OLS", n_sub=4, custom="all", plots=())   # plots=() -> no rendering
        # pooled, pooled+covariate, T1, T2
        self.assertEqual(len(results), 4)
        self.assertIn("all cell types (pooled)", results)
        self.assertTrue(any("covariate" in k for k in results))
        self.assertIn("T1 only", results)
        # a cancer custom factor shows up as a predictor row somewhere.
        any_custom = any(
            any(str(f).endswith("-CANCER") or str(f).endswith("-proximity")
                for f in res["factor"])
            for res in results.values())
        self.assertTrue(any_custom)


    def test_run_one_single_fit(self):
        adata = _make_store_adata(2)
        res = sh.run_one(adata, "TARGET", {"TARGET": ["A", "B"]},
                         annot_col="cell_type", annot_value="T1", onehot_col=None,
                         method="OLS", n_sub=4, custom="all", plots=())
        self.assertIsNotNone(res)
        self.assertIn("avg_subsample_beta", res.columns)


class PlotRenderTests(unittest.TestCase):
    def test_ratio_plots_render(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        res = pd.DataFrame({
            "factor": ["TF1", "TF2", "TF3"], "beta": [1.0, -2.0, 0.5],
            "min_subsample_beta": [0.5, -2.5, 0.1], "max_subsample_beta": [1.5, -1.5, 0.9],
            "avg_subsample_beta": [1.0, -2.0, 0.5], "var_subsample_beta": [0.1, 0.2, 0.05],
            "meansq_subsample_beta": [1.1, 4.2, 0.3], "r": [0.3, -0.4, 0.1],
        })
        ax = sh.filter_and_plot_meansq_variance_ratio(res, highlight_genes=["TF1"])
        self.assertIsNotNone(ax)
        sh.plot_res(res, highlight_genes=["TF1"], title="t")   # all 3 ratio plots; must not raise
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
