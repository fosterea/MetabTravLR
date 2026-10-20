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
