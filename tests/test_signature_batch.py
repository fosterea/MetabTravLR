"""Tests for `metab_processing/LinearRegression/signature_batch.py` -- the shared compute
surface for the signature-analysis cache + SLURM dispatch. Synthetic data only (no network,
no `load_msigdb`); `compute_dataset` is NOT tested end-to-end here since it calls the
network `load_msigdb` -- see the module docstring / task spec."""
import os
import shutil
import sys
import tempfile
import unittest
import warnings
from unittest import mock

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import anndata as ad

from metab_processing.LinearRegression import signature_batch as sb
from metab_processing.LinearRegression.signature_configs import DatasetConfig

N = 80
SIG_GENES = ["A", "B", "C", "D"]
OTHER = ["TF1", "TF2", "LIG", "REC"]
GENES = SIG_GENES + OTHER


def _make_store_adata(seed=0):
    """Same fixture shape as `tests/test_signature_helpers.py::_make_store_adata`: a tiny
    adata with a hand-built factor store + spatial + a cell_type with two T subtypes and an
    'other' type, so pooled / covariate / per-type run specs all have cells to fit on."""
    rng = np.random.default_rng(seed)
    X = (rng.random((N, len(GENES))).astype(np.float32) + 0.1)
    a = ad.AnnData(X=X.copy())
    a.var_names = GENES
    a.obs_names = [f"c{i}" for i in range(N)]
    a.layers["normalized_count"] = X.copy()
    a.obsm["spatial"] = rng.uniform(0, 300, size=(N, 2))
    cts = np.array(["T1", "T2", "other"])[rng.integers(0, 3, size=N)]
    a.obs["cell_type"] = pd.Categorical(cts)

    cols = ["TF1", "TF2", "B"]
    a.obsm["x_factors"] = X[:, [GENES.index("TF1"), GENES.index("TF2"), GENES.index("B")]].copy()
    a.uns["x_factors_cols"] = cols
    a.uns["x_factor_map"] = {"A": ["TF1", "B"], "B": ["TF1"], "C": ["TF2"], "D": ["TF2"]}
    a.uns["x_genes"] = ["A", "B", "C", "D"]
    return a


def _minimal_cfg(**overrides):
    kwargs = dict(
        name="test", kind="cancer", adata_path="", cell_type_col="cell_type",
        target_signatures=["TARGET"], neg_controls=[],
        t_cell_types=["T1", "T2"], cancer_factors=[], vessel={},
        method="OLS", penalty=1.0, standardize=False, n_sub=4, lower=15, upper=85,
    )
    kwargs.update(overrides)
    return DatasetConfig(**kwargs)


class RunspecsCancerTests(unittest.TestCase):
    def test_two_t_cell_types_has_pooled_and_cov_and_per_type(self):
        cfg = _minimal_cfg(t_cell_types=["T1", "T2"])
        specs = sb.runspecs(cfg)
        labels = [s["label"] for s in specs]
        self.assertIn("all_pooled", labels)
        self.assertIn("all_pooled_cov_cell_type", labels)
        self.assertIn("T1_only", labels)
        self.assertIn("T2_only", labels)
        self.assertEqual(len(specs), 4)

        pooled = next(s for s in specs if s["label"] == "all_pooled")
        self.assertEqual(pooled["annot_col"], "_tcell_group")
        self.assertEqual(pooled["annot_value"], "GROUP")
        self.assertIsNone(pooled["onehot_col"])
        self.assertIsNone(pooled["subset"])

        cov = next(s for s in specs if s["label"] == "all_pooled_cov_cell_type")
        self.assertEqual(cov["annot_col"], "_tcell_group")
        self.assertEqual(cov["onehot_col"], "cell_type")

        t1 = next(s for s in specs if s["label"] == "T1_only")
        self.assertEqual(t1["annot_col"], "cell_type")
        self.assertEqual(t1["annot_value"], "T1")
        self.assertIsNone(t1["onehot_col"])

    def test_single_t_cell_type_has_only_per_type_run(self):
        cfg = _minimal_cfg(t_cell_types=["T1"])
        specs = sb.runspecs(cfg)
        labels = [s["label"] for s in specs]
        self.assertEqual(labels, ["T1_only"])
        self.assertNotIn("all_pooled", labels)


class RunspecsColonTests(unittest.TestCase):
    def _colon_cfg(self):
        return _minimal_cfg(
            kind="colon", t_cell_types=[], target_signatures=["TARGET"], neg_controls=[],
            annot_col="cell_type", cell_type="T1",
            condition_col="condition", patient_col="patient", sample_col="sample",
            single_gene="A",
        )

    def test_colon_grid_labels_and_subsets(self):
        cfg = self._colon_cfg()
        specs = sb.runspecs(cfg)
        # 4 "all" covariate runs + 3 conditions x 3 covariates = 13
        self.assertEqual(len(specs), 13)
        labels = [s["label"] for s in specs]
        for expected in ("all__cov_none", "all__cov_condition", "all__cov_patient",
                         "all__cov_sample"):
            self.assertIn(expected, labels)
        for cond in ("UC", "HC", "ICI"):
            for expected in (f"{cond}__cov_none", f"{cond}__cov_patient",
                             f"{cond}__cov_sample"):
                self.assertIn(expected, labels)

        all_none = next(s for s in specs if s["label"] == "all__cov_none")
        self.assertEqual(all_none["annot_col"], "cell_type")
        self.assertEqual(all_none["annot_value"], "T1")
        self.assertIsNone(all_none["onehot_col"])
        self.assertIsNone(all_none["subset"])

        all_cond = next(s for s in specs if s["label"] == "all__cov_condition")
        self.assertEqual(all_cond["onehot_col"], "condition")
        self.assertIsNone(all_cond["subset"])

        uc_patient = next(s for s in specs if s["label"] == "UC__cov_patient")
        self.assertEqual(uc_patient["onehot_col"], "patient")
        self.assertEqual(uc_patient["subset"], ("condition", "UC"))


class SlugAndCachePathTests(unittest.TestCase):
    def test_slug_strips_specials_and_edges(self):
        self.assertEqual(sb._slug("HALLMARK_IL2 STAT5!!"), "HALLMARK_IL2_STAT5")
        self.assertEqual(sb._slug("  weird//name--here  "), "weird_name_here")
        self.assertEqual(sb._slug("T1_only"), "T1_only")

    def test_cache_path_puts_file_under_params_subdir(self):
        cfg = _minimal_cfg(method="l1", penalty=0.005, standardize=True)
        p = sb.cache_path(cfg, "HALLMARK_FOO", "T1_only")
        self.assertEqual(p.parent.parent.as_posix(), cfg.cache_dir)
        self.assertEqual(p.parent.name, sb.params_slug("l1", 0.005, cfg.l1_ratio, True))
        self.assertEqual(p.name, "HALLMARK_FOO__T1_only.parquet")


class NumAndParamsSlugTests(unittest.TestCase):
    def test_num_matches_regardless_of_literal_form(self):
        self.assertEqual(sb._num(0.005), sb._num(5e-3))
        self.assertEqual(sb._num(0.005), "0p005")

    def test_num_handles_scientific_and_negative(self):
        self.assertEqual(sb._num(1e-05), "1em05")
        self.assertEqual(sb._num(-0.5), "m0p5")

    def test_params_slug_basic_l1(self):
        self.assertEqual(sb.params_slug("l1", 0.005, 0.5, True), "l1_p0p005")

    def test_params_slug_elastic_adds_l1_ratio(self):
        slug = sb.params_slug("elastic", 1.0, 0.3, True)
        self.assertEqual(slug, "elastic_p1_l0p3")

    def test_params_slug_l1_ratio_ignored_for_non_elastic(self):
        # l1_ratio must not affect the slug for a non-elastic method.
        a = sb.params_slug("l1", 0.5, 0.1, True)
        b = sb.params_slug("l1", 0.5, 0.9, True)
        self.assertEqual(a, b)

    def test_params_slug_nostd_suffix(self):
        self.assertEqual(sb.params_slug("l1", 0.005, 0.5, False), "l1_p0p005_nostd")
        self.assertEqual(sb.params_slug("l1", 0.005, 0.5, True), "l1_p0p005")

    def test_params_slug_is_deterministic_for_equal_values(self):
        self.assertEqual(sb.params_slug("l1", 0.005, 0.5, True),
                         sb.params_slug("l1", 5e-3, 0.5, True))


class BetasPathAndSkipMarkerTests(unittest.TestCase):
    def test_betas_path_shape(self):
        cfg = _minimal_cfg(method="l1", penalty=0.005)
        res_p = sb.cache_path(cfg, "TARGET", "T1_only")
        betas_p = sb.betas_path(cfg, "TARGET", "T1_only")
        self.assertEqual(betas_p.parent, res_p.parent)
        self.assertEqual(betas_p.name, "TARGET__T1_only.betas.parquet")

    def test_skip_marker_shares_params_subdir_with_cache_path(self):
        cfg = _minimal_cfg(method="l1", penalty=0.005)
        res_p = sb.cache_path(cfg, "TARGET", "T1_only")
        skip_p = sb._skip_marker(cfg, "TARGET", "T1_only")
        self.assertEqual(skip_p.parent, res_p.parent)
        self.assertEqual(skip_p.suffix, ".skip")

    def test_overrides_resolve_to_a_different_subdir(self):
        cfg = _minimal_cfg(method="l1", penalty=0.005)
        default_p = sb.cache_path(cfg, "TARGET", "T1_only")
        override_p = sb.cache_path(cfg, "TARGET", "T1_only", method="ridge", penalty=1.0)
        self.assertNotEqual(default_p.parent, override_p.parent)


class LoadResTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.cfg = _minimal_cfg(cache_root=self.tmp)

    def test_missing_file_returns_none(self):
        self.assertIsNone(sb.load_res(self.cfg, "NOPE", "T1_only"))

    def test_round_trips_a_written_parquet(self):
        df = pd.DataFrame({"factor": ["TF1", "TF2"], "beta": [0.5, -1.5]})
        p = sb.cache_path(self.cfg, "TARGET", "T1_only")
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p)
        out = sb.load_res(self.cfg, "TARGET", "T1_only")
        pd.testing.assert_frame_equal(out, df)

    def test_load_betas_missing_returns_none(self):
        self.assertIsNone(sb.load_betas(self.cfg, "NOPE", "T1_only"))

    def test_load_betas_round_trips(self):
        df = pd.DataFrame({"TF1": [0.1, 0.2], "TF2": [-0.3, -0.4]})
        p = sb.betas_path(self.cfg, "TARGET", "T1_only")
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p)
        out = sb.load_betas(self.cfg, "TARGET", "T1_only")
        pd.testing.assert_frame_equal(out, df)


class ComputeRunTests(unittest.TestCase):
    def setUp(self):
        self.adata = _make_store_adata(5)
        self.cfg = _minimal_cfg(method="OLS", n_sub=3)

    def test_returns_expected_columns(self):
        spec = {"annot_col": "cell_type", "annot_value": "T1", "onehot_col": None,
                "subset": None, "label": "t"}
        res, betas = sb.compute_run(self.adata, "TARGET", ["A", "B"], spec, self.cfg, raw=False)
        self.assertFalse(res.empty)
        for col in ("gene", "factor", "beta", "r", "r2", "model_r2", "n_cells",
                    "min_subsample_beta", "max_subsample_beta", "avg_subsample_beta",
                    "meansq_subsample_beta", "msd_subsample_beta"):
            self.assertIn(col, res.columns)

    def test_returns_tuple_with_raw_betas_shape(self):
        """`betas` is the raw `n_sub x n_selected_factors` subsample-beta frame (the `sub`
        `subsample_gene_betas(..., return_r=True)` builds), distinct from the ranked/
        percentiled `res` table."""
        spec = {"annot_col": "cell_type", "annot_value": "T1", "onehot_col": None,
                "subset": None, "label": "t"}
        res, betas = sb.compute_run(self.adata, "TARGET", ["A", "B"], spec, self.cfg, raw=False)
        self.assertIsInstance(betas, pd.DataFrame)
        self.assertEqual(betas.shape[0], self.cfg.n_sub)
        self.assertEqual(betas.shape[1], len(res))     # one column per selected factor
        self.assertEqual(set(betas.columns), set(res["factor"]))

    def test_subset_spec_selects_fewer_cells(self):
        t1_idx = np.where(self.adata.obs["cell_type"] == "T1")[0]
        self.assertGreater(len(t1_idx), 4, "fixture must have >4 T1 cells for this test")
        half = len(t1_idx) // 2
        batch = np.array(["y"] * N, dtype=object)
        batch[t1_idx[:half]] = "x"
        self.adata.obs["batch"] = batch

        spec_full = {"annot_col": "cell_type", "annot_value": "T1", "onehot_col": None,
                     "subset": None, "label": "full"}
        spec_sub = {"annot_col": "cell_type", "annot_value": "T1", "onehot_col": None,
                    "subset": ("batch", "x"), "label": "sub"}

        res_full, _ = sb.compute_run(self.adata, "TARGET", ["A", "B"], spec_full, self.cfg)
        res_sub, _ = sb.compute_run(self.adata, "TARGET", ["A", "B"], spec_sub, self.cfg)

        self.assertEqual(int(res_full["n_cells"].iloc[0]), len(t1_idx))
        self.assertEqual(int(res_sub["n_cells"].iloc[0]), half)
        self.assertLess(res_sub["n_cells"].iloc[0], res_full["n_cells"].iloc[0])

    def test_raw_true_fits_the_raw_gene(self):
        spec = {"annot_col": "cell_type", "annot_value": "T1", "onehot_col": None,
                "subset": None, "label": "t"}
        res, betas = sb.compute_run(self.adata, "B", [], spec, self.cfg, raw=True)
        self.assertFalse(res.empty)
        self.assertEqual(set(res["gene"]), {"B"})
        self.assertFalse(betas.empty)


class ComputeDatasetShortCircuitTests(unittest.TestCase):
    """The resume contract: a dataset with a permanently-missing set (or a failing combo)
    must still SHORT-CIRCUIT on the second call -- i.e. not reload the adata / re-diffuse."""

    def test_short_circuits_despite_missing_set(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        adata = _make_store_adata(7)
        adata_path = os.path.join(tmp, "adata.h5ad")
        adata.write_h5ad(adata_path)

        cfg = _minimal_cfg(name="sc", kind="cancer", adata_path=adata_path, cache_root=tmp,
                           t_cell_types=["T1"],                 # single T type -> one spec
                           target_signatures=["PRESENT", "MISSING"], neg_controls=[],
                           cancer_factors=[], vessel={}, method="OLS", n_sub=3)
        library = {"PRESENT": ["A", "B", "C"]}                  # MISSING absent from MSigDB

        reads = {"n": 0}
        real_read = sb.sc.read_h5ad

        def counting_read(path, *a, **k):
            reads["n"] += 1
            return real_read(path, *a, **k)

        with mock.patch.object(sb.sh, "load_msigdb", return_value=library), \
             mock.patch.object(sb.sc, "read_h5ad", side_effect=counting_read):
            sb.compute_dataset(cfg)
            self.assertEqual(reads["n"], 1)                      # loaded once
            self.assertIsNotNone(sb.load_res(cfg, "PRESENT", "T1_only"))
            self.assertTrue(sb._skip_marker(cfg, "MISSING", "T1_only").exists())

            # Both the result parquet AND its raw-betas sidecar were written, under the
            # same settings subdir; load_betas round-trips what compute_run returned.
            res_path = sb.cache_path(cfg, "PRESENT", "T1_only")
            betas_path = sb.betas_path(cfg, "PRESENT", "T1_only")
            self.assertTrue(res_path.exists())
            self.assertTrue(betas_path.exists())
            self.assertEqual(betas_path.parent, res_path.parent)
            betas_df = sb.load_betas(cfg, "PRESENT", "T1_only")
            self.assertIsNotNone(betas_df)
            res_df = sb.load_res(cfg, "PRESENT", "T1_only")
            self.assertEqual(set(betas_df.columns), set(res_df["factor"]))
            self.assertEqual(len(betas_df), cfg.n_sub)

            sb.compute_dataset(cfg)                              # second call
            self.assertEqual(reads["n"], 1)                      # short-circuited: no reload

    def test_overwrite_then_fail_clears_stale_cache(self):
        """overwrite=True on a now-FAILING combo must remove the prior success's res+betas
        (not leave them next to a .skip marker, or a later run serves stale data)."""
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        adata = _make_store_adata(11)
        adata_path = os.path.join(tmp, "adata.h5ad")
        adata.write_h5ad(adata_path)
        cfg = _minimal_cfg(name="sf", kind="cancer", adata_path=adata_path, cache_root=tmp,
                           t_cell_types=["T1"], target_signatures=["PRESENT"], neg_controls=[],
                           cancer_factors=[], vessel={}, method="OLS", n_sub=3)
        lib = {"PRESENT": ["A", "B", "C"]}

        with mock.patch.object(sb.sh, "load_msigdb", return_value=lib):
            sb.compute_dataset(cfg)
        self.assertIsNotNone(sb.load_res(cfg, "PRESENT", "T1_only"))
        self.assertIsNotNone(sb.load_betas(cfg, "PRESENT", "T1_only"))

        with mock.patch.object(sb.sh, "load_msigdb", return_value=lib), \
             mock.patch.object(sb, "compute_run", side_effect=RuntimeError("boom")):
            sb.compute_dataset(cfg, overwrite=True)
        # stale res + betas gone; only the skip marker remains
        self.assertIsNone(sb.load_res(cfg, "PRESENT", "T1_only"))
        self.assertIsNone(sb.load_betas(cfg, "PRESENT", "T1_only"))
        self.assertTrue(sb._skip_marker(cfg, "PRESENT", "T1_only").exists())

        with mock.patch.object(sb.sh, "load_msigdb", return_value=lib):
            sb.compute_dataset(cfg)                              # plain rerun short-circuits
        self.assertIsNone(sb.load_res(cfg, "PRESENT", "T1_only"))   # NOT resurrected


class ManualSetTests(unittest.TestCase):
    """A manual (non-MSigDB) target set is resolved from its literal gene list, computed
    (not treated as missing), and appears in membership with the factor-split report."""

    def test_manual_set_is_a_target_and_computed(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        adata = _make_store_adata(9)
        adata_path = os.path.join(tmp, "adata.h5ad")
        adata.write_h5ad(adata_path)

        cfg = _minimal_cfg(name="man", kind="cancer", adata_path=adata_path, cache_root=tmp,
                           t_cell_types=["T1"], target_signatures=["PRESENT"], neg_controls=[],
                           cancer_factors=[], vessel={}, method="OLS", n_sub=3,
                           manual_sets={"MAN_NEG": ["A", "B"]})   # literal genes, in var
        self.assertIn("MAN_NEG", sb.all_targets(cfg))

        with mock.patch.object(sb.sh, "load_msigdb", return_value={"PRESENT": ["A", "B", "C"]}):
            sb.compute_dataset(cfg)

        import json
        membership = json.load(open(os.path.join(cfg.cache_dir, "_membership.json")))
        self.assertIn("MAN_NEG", membership)
        self.assertEqual(set(membership["MAN_NEG"]["present"]), {"A", "B"})
        self.assertIn("factor_removed", membership["MAN_NEG"])     # treated as a target
        # computed (not skipped as "missing from MSigDB")
        self.assertIsNotNone(sb.load_res(cfg, "MAN_NEG", "T1_only"))
        self.assertFalse(sb._skip_marker(cfg, "MAN_NEG", "T1_only").exists())


if __name__ == "__main__":
    unittest.main()
