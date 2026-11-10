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


class ComputeDatasetTargetsSubsetTests(unittest.TestCase):
    """`compute_dataset(cfg, targets=...)` restricts the compute loop to a subset of
    `all_targets(cfg)` -- the mechanism behind fanning one dataset across concurrent jobs.
    Disjoint slices must (a) each compute only their own targets and (b) together reproduce
    exactly the full-run cache."""

    def _setup(self, seed):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        adata = _make_store_adata(seed)
        adata_path = os.path.join(tmp, "adata.h5ad")
        adata.write_h5ad(adata_path)
        cfg = _minimal_cfg(name="sub", kind="cancer", adata_path=adata_path, cache_root=tmp,
                           t_cell_types=["T1"],                    # single T type -> one spec
                           target_signatures=["S1", "S2", "S3"], neg_controls=[],
                           cancer_factors=[], vessel={}, method="OLS", n_sub=3)
        library = {"S1": ["A", "B", "C"], "S2": ["A", "B", "D"], "S3": ["B", "C", "D"]}
        return cfg, library

    def test_targets_restricts_compute_to_the_subset(self):
        cfg, library = self._setup(21)
        with mock.patch.object(sb.sh, "load_msigdb", return_value=library):
            sb.compute_dataset(cfg, targets=["S2"])
        # Only S2 computed; S1/S3 left entirely uncomputed (no res, no skip marker).
        self.assertIsNotNone(sb.load_res(cfg, "S2", "T1_only"))
        for other in ("S1", "S3"):
            self.assertIsNone(sb.load_res(cfg, other, "T1_only"))
            self.assertFalse(sb._skip_marker(cfg, other, "T1_only").exists())
        # Membership is still written in FULL (every job needs the complete sidecar).
        import json
        membership = json.load(open(os.path.join(cfg.cache_dir, "_membership.json")))
        self.assertEqual({"S1", "S2", "S3"}, set(membership))

    def test_unknown_target_raises(self):
        cfg, library = self._setup(22)
        with mock.patch.object(sb.sh, "load_msigdb", return_value=library):
            with self.assertRaises(ValueError):
                sb.compute_dataset(cfg, targets=["S2", "NOPE"])

    def test_duplicate_and_unordered_targets_dedupe(self):
        cfg, library = self._setup(24)
        with mock.patch.object(sb.sh, "load_msigdb", return_value=library):
            # reversed order + a duplicate -> deduped, both computed, no error
            sb.compute_dataset(cfg, targets=["S2", "S1", "S2"])
        self.assertIsNotNone(sb.load_res(cfg, "S1", "T1_only"))
        self.assertIsNotNone(sb.load_res(cfg, "S2", "T1_only"))
        self.assertIsNone(sb.load_res(cfg, "S3", "T1_only"))

    def test_neg_control_target_in_slice(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        adata = _make_store_adata(25)
        adata_path = os.path.join(tmp, "adata.h5ad")
        adata.write_h5ad(adata_path)
        cfg = _minimal_cfg(name="nc", kind="cancer", adata_path=adata_path, cache_root=tmp,
                           t_cell_types=["T1"], target_signatures=["S1"],
                           neg_controls=["NC"], cancer_factors=[], vessel={},
                           method="OLS", n_sub=3)
        library = {"S1": ["A", "B", "C"], "NC": ["B", "C", "D"]}
        # A neg_control is a valid target name (via all_targets) and must be selectable.
        self.assertIn("NC", sb.all_targets(cfg))
        with mock.patch.object(sb.sh, "load_msigdb", return_value=library):
            sb.compute_dataset(cfg, targets=["NC"])
        self.assertIsNotNone(sb.load_res(cfg, "NC", "T1_only"))
        self.assertIsNone(sb.load_res(cfg, "S1", "T1_only"))

    def test_empty_targets_is_a_noop_without_loading_adata(self):
        cfg, library = self._setup(26)
        reads = {"n": 0}
        real_read = sb.sc.read_h5ad

        def counting_read(path, *a, **k):
            reads["n"] += 1
            return real_read(path, *a, **k)

        with mock.patch.object(sb.sh, "load_msigdb", return_value=library), \
             mock.patch.object(sb.sc, "read_h5ad", side_effect=counting_read):
            sb.compute_dataset(cfg, targets=[])          # empty selection
        self.assertEqual(reads["n"], 0)                   # returned before any adata load
        for name in ("S1", "S2", "S3"):
            self.assertIsNone(sb.load_res(cfg, name, "T1_only"))
        self.assertFalse(os.path.exists(os.path.join(cfg.cache_dir, "_membership.json")))

    def test_collision_guard_runs_over_full_list_not_just_slice(self):
        # Two target names that _slug to the SAME cache filename but sit in DIFFERENT slices.
        # The guard must still raise, or two fan-out jobs would overwrite each other's file.
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        adata = _make_store_adata(27)
        adata_path = os.path.join(tmp, "adata.h5ad")
        adata.write_h5ad(adata_path)
        cfg = _minimal_cfg(name="col", kind="cancer", adata_path=adata_path, cache_root=tmp,
                           t_cell_types=["T1"], target_signatures=["A-B", "A_B"],  # both -> 'A_B'
                           neg_controls=[], cancer_factors=[], vessel={},
                           method="OLS", n_sub=3)
        # Select only ONE of the colliding pair; the guard over the full list still fires.
        with self.assertRaises(ValueError):
            sb.compute_dataset(cfg, targets=["A-B"])

    def test_disjoint_slices_reproduce_full_run(self):
        # Full run into one cache dir.
        cfg_full, library = self._setup(23)
        with mock.patch.object(sb.sh, "load_msigdb", return_value=library):
            sb.compute_dataset(cfg_full)

        # Two disjoint slices into a SEPARATE cache dir (same adata/library/seed).
        cfg_a, lib_a = self._setup(23)
        with mock.patch.object(sb.sh, "load_msigdb", return_value=lib_a):
            sb.compute_dataset(cfg_a, targets=["S1"])
            sb.compute_dataset(cfg_a, targets=["S2", "S3"])

        # Same set of result files, and each (target) result is numerically identical.
        for name in ("S1", "S2", "S3"):
            full = sb.load_res(cfg_full, name, "T1_only")
            split = sb.load_res(cfg_a, name, "T1_only")
            self.assertIsNotNone(split, f"{name} missing from the split run")
            pd.testing.assert_frame_equal(full, split)


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


class SweepMethodSlugAndPathTests(unittest.TestCase):
    def test_sweep_method_slug_basic(self):
        self.assertEqual(sb.sweep_method_slug("l1", 0.5, True), "l1")
        self.assertEqual(sb.sweep_method_slug("ridge", 0.5, True), "ridge")

    def test_sweep_method_slug_elastic_adds_l1_ratio(self):
        self.assertEqual(sb.sweep_method_slug("elastic", 0.3, True), "elastic_l0p3")

    def test_sweep_method_slug_nostd_suffix(self):
        self.assertEqual(sb.sweep_method_slug("l1", 0.5, False), "l1_nostd")

    def test_sweep_method_slug_has_no_penalty_component(self):
        # Unlike params_slug, the penalty never appears in a sweep slug (penalties live
        # inside the cached file, not the path).
        self.assertNotIn("p", sb.sweep_method_slug("l1", 0.5, True))

    def test_sweep_path_under_sweep_method_subdir(self):
        cfg = _minimal_cfg(method="l1")
        p = sb.sweep_path(cfg, "HALLMARK_FOO")
        self.assertEqual(p.parent.parent.name, "sweep")
        self.assertEqual(p.parent.name, sb.sweep_method_slug("l1", cfg.l1_ratio, cfg.standardize))
        self.assertEqual(p.name, "HALLMARK_FOO.parquet")

    def test_sweep_path_method_override(self):
        cfg = _minimal_cfg(method="l1", standardize=True)
        p_l1 = sb.sweep_path(cfg, "T", method="l1")
        p_ridge = sb.sweep_path(cfg, "T", method="ridge")
        self.assertNotEqual(p_l1.parent, p_ridge.parent)
        self.assertEqual(p_ridge.parent.name, "ridge")


class LoadSweepTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.cfg = _minimal_cfg(cache_root=self.tmp)

    def test_missing_returns_none(self):
        self.assertIsNone(sb.load_sweep(self.cfg, "NOPE"))

    def test_round_trips_a_written_parquet(self):
        # subsample mixes the string 'full' with int ids -- pyarrow can't write that
        # column as-is (see `_encode_sweep_subsample`), so cache writes go through the
        # encode helper; `load_sweep` must hand back the original mixed-type contract.
        df = pd.DataFrame({"subsample": ["full", 0], "penalty": [0.1, 0.1],
                           "factor": ["TF1", "TF1"], "beta": [0.5, 0.4], "n_cells": [10, 10]})
        p = sb.sweep_path(self.cfg, "TARGET", method=self.cfg.method)
        sb._atomic_write(p, df=sb._encode_sweep_subsample(df))
        out = sb.load_sweep(self.cfg, "TARGET", method=self.cfg.method)
        pd.testing.assert_frame_equal(out, df)


class SweepCellTypesTests(unittest.TestCase):
    def test_defaults_to_t_cell_types_for_cancer(self):
        cfg = _minimal_cfg(t_cell_types=["T1", "T2"], sweep_cell_types=[])
        self.assertEqual(sb._sweep_cell_types(cfg), ["T1", "T2"])

    def test_explicit_sweep_cell_types_override(self):
        cfg = _minimal_cfg(t_cell_types=["T1", "T2"], sweep_cell_types=["T1"])
        self.assertEqual(sb._sweep_cell_types(cfg), ["T1"])

    def test_defaults_to_cell_type_for_colon(self):
        cfg = _minimal_cfg(kind="colon", t_cell_types=[], sweep_cell_types=[],
                           annot_col="cell_type", cell_type="T1", target_signatures=["T"])
        self.assertEqual(sb._sweep_cell_types(cfg), ["T1"])


class ComputeSweepDatasetIncrementalTests(unittest.TestCase):
    """The incremental-cache contract: re-running with MORE penalties in the config must
    only compute the NEW ones and leave a UNION of old+new penalties cached -- never
    re-pay (or drop) what was already there."""

    def _cfg(self, tmp, adata_path, penalties):
        return _minimal_cfg(
            name="sweep_test", kind="cancer", adata_path=adata_path, cache_root=tmp,
            t_cell_types=["T1"], target_signatures=["PRESENT"], neg_controls=[],
            cancer_factors=[], vessel={}, method="l1", standardize=True,
            sweep_penalties={"l1": penalties}, sweep_k=2,
        )

    def test_extending_penalties_only_computes_the_new_one(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        adata = _make_store_adata(13)
        adata_path = os.path.join(tmp, "adata.h5ad")
        adata.write_h5ad(adata_path)
        library = {"PRESENT": ["A", "B", "C"]}

        cfg_ab = self._cfg(tmp, adata_path, [0.1, 0.5])
        with mock.patch.object(sb.sh, "load_msigdb", return_value=library):
            sb.compute_sweep_dataset(cfg_ab, "l1")

        cached = sb.load_sweep(cfg_ab, "PRESENT", method="l1")
        self.assertIsNotNone(cached)
        self.assertEqual(set(cached["penalty"].unique()), {0.1, 0.5})
        self.assertEqual(set(cached["subsample"].unique()), {"full", 0, 1})

        # Extend the config to add a third penalty and re-run: only 0.9 should be new,
        # and the resulting cache must hold the UNION {0.1, 0.5, 0.9}.
        cfg_abc = self._cfg(tmp, adata_path, [0.1, 0.5, 0.9])
        with mock.patch.object(sb.sh, "load_msigdb", return_value=library), \
             mock.patch.object(sb.ls, "sweep_subsample_betas",
                               side_effect=sb.ls.sweep_subsample_betas) as spy:
            sb.compute_sweep_dataset(cfg_abc, "l1")
            # exactly one call, asked for exactly the missing penalty.
            self.assertEqual(spy.call_count, 1)
            _, kwargs_or_args = spy.call_args, None
            called_penalties = spy.call_args[0][2]
            self.assertEqual(list(called_penalties), [0.9])

        merged = sb.load_sweep(cfg_abc, "PRESENT", method="l1")
        self.assertEqual(set(merged["penalty"].unique()), {0.1, 0.5, 0.9})
        self.assertEqual(set(merged["subsample"].unique()), {"full", 0, 1})
        # old rows preserved byte-for-byte (not recomputed): same number of (subsample,
        # factor) rows at penalty 0.1 before and after.
        old_01 = cached[cached["penalty"] == 0.1].reset_index(drop=True)
        new_01 = merged[merged["penalty"] == 0.1].reset_index(drop=True)
        pd.testing.assert_frame_equal(
            old_01.sort_values(["subsample", "factor"]).reset_index(drop=True),
            new_01.sort_values(["subsample", "factor"]).reset_index(drop=True))

    def test_second_call_same_penalties_computes_nothing_new(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        adata = _make_store_adata(17)
        adata_path = os.path.join(tmp, "adata.h5ad")
        adata.write_h5ad(adata_path)
        library = {"PRESENT": ["A", "B", "C"]}
        cfg = self._cfg(tmp, adata_path, [0.1, 0.5])

        with mock.patch.object(sb.sh, "load_msigdb", return_value=library):
            sb.compute_sweep_dataset(cfg, "l1")
            with mock.patch.object(sb.ls, "sweep_subsample_betas") as spy:
                sb.compute_sweep_dataset(cfg, "l1")
                spy.assert_not_called()

    def test_unresolvable_target_is_skipped(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        adata = _make_store_adata(19)
        adata_path = os.path.join(tmp, "adata.h5ad")
        adata.write_h5ad(adata_path)
        cfg = _minimal_cfg(
            name="sweep_missing", kind="cancer", adata_path=adata_path, cache_root=tmp,
            t_cell_types=["T1"], target_signatures=["PRESENT", "MISSING"], neg_controls=[],
            cancer_factors=[], vessel={}, method="l1", standardize=True,
            sweep_penalties={"l1": [0.1]}, sweep_k=2,
        )
        library = {"PRESENT": ["A", "B", "C"]}   # MISSING absent from MSigDB
        with mock.patch.object(sb.sh, "load_msigdb", return_value=library):
            sb.compute_sweep_dataset(cfg, "l1")
        self.assertIsNotNone(sb.load_sweep(cfg, "PRESENT", method="l1"))
        self.assertIsNone(sb.load_sweep(cfg, "MISSING", method="l1"))
        # an unresolvable target leaves a .skip marker so the next run short-circuits.
        self.assertTrue(sb._sweep_skip_marker(cfg, "MISSING", method="l1").exists())


class ComputeSweepDatasetResumabilityTests(unittest.TestCase):
    """The SLURM-resumability contract mirrored from `compute_dataset`: a target whose fit
    RAISES must be `.skip`-marked (not abort the run), a fully-cached/skip-marked dataset
    must short-circuit BEFORE reading adata, and `overwrite=True` must clear a stale skip."""

    def _cancer_cfg(self, tmp, adata_path, targets, penalties=(0.1,)):
        return _minimal_cfg(
            name="sweep_resume", kind="cancer", adata_path=adata_path, cache_root=tmp,
            t_cell_types=["T1"], target_signatures=list(targets), neg_controls=[],
            cancer_factors=[], vessel={}, method="l1", standardize=True,
            sweep_penalties={"l1": list(penalties)}, sweep_k=2,
        )

    def _write_adata(self, tmp, seed):
        adata = _make_store_adata(seed)
        p = os.path.join(tmp, "adata.h5ad")
        adata.write_h5ad(p)
        return p

    def test_raising_target_is_skip_marked_and_others_still_computed(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        adata_path = self._write_adata(tmp, 21)
        cfg = self._cancer_cfg(tmp, adata_path, ["PRESENT", "BOOM"])
        library = {"PRESENT": ["A", "B", "C"], "BOOM": ["A", "B", "C"]}

        real = sb.ls.sweep_subsample_betas

        def fake(ad2, gene, pens, **kw):
            if gene == "BOOM":
                raise ValueError("boom")
            return real(ad2, gene, pens, **kw)

        with mock.patch.object(sb.sh, "load_msigdb", return_value=library), \
             mock.patch.object(sb.ls, "sweep_subsample_betas", side_effect=fake):
            sb.compute_sweep_dataset(cfg, "l1")          # must NOT raise

        self.assertIsNotNone(sb.load_sweep(cfg, "PRESENT", method="l1"))
        self.assertIsNone(sb.load_sweep(cfg, "BOOM", method="l1"))
        self.assertTrue(sb._sweep_skip_marker(cfg, "BOOM", method="l1").exists())

    def test_fully_cached_run_short_circuits_before_reading_adata(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        adata_path = self._write_adata(tmp, 23)
        cfg = self._cancer_cfg(tmp, adata_path, ["PRESENT"])
        library = {"PRESENT": ["A", "B", "C"]}

        with mock.patch.object(sb.sh, "load_msigdb", return_value=library):
            sb.compute_sweep_dataset(cfg, "l1")

        # Second run: everything cached -> must not touch adata / MSigDB at all.
        with mock.patch.object(sb.sc, "read_h5ad",
                               side_effect=AssertionError("adata re-read")) as rd, \
             mock.patch.object(sb.sh, "load_msigdb",
                               side_effect=AssertionError("MSigDB re-read")):
            sb.compute_sweep_dataset(cfg, "l1")
            rd.assert_not_called()

    def test_skip_marked_dataset_short_circuits(self):
        # A dataset whose ONLY target is permanently unresolvable must still short-circuit
        # (skip marker counts as 'done') and not re-read adata on the next run.
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        adata_path = self._write_adata(tmp, 24)
        cfg = self._cancer_cfg(tmp, adata_path, ["MISSING"])
        with mock.patch.object(sb.sh, "load_msigdb", return_value={}):
            sb.compute_sweep_dataset(cfg, "l1")
        self.assertTrue(sb._sweep_skip_marker(cfg, "MISSING", method="l1").exists())
        with mock.patch.object(sb.sc, "read_h5ad",
                               side_effect=AssertionError("adata re-read")) as rd:
            sb.compute_sweep_dataset(cfg, "l1")
            rd.assert_not_called()

    def test_overwrite_clears_stale_skip_and_recomputes(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        adata_path = self._write_adata(tmp, 25)
        cfg = self._cancer_cfg(tmp, adata_path, ["BOOM"])
        library = {"BOOM": ["A", "B", "C"]}
        real = sb.ls.sweep_subsample_betas

        # First run raises -> skip marker written, no cache.
        def boom(ad2, gene, pens, **kw):
            raise ValueError("boom")

        with mock.patch.object(sb.sh, "load_msigdb", return_value=library), \
             mock.patch.object(sb.ls, "sweep_subsample_betas", side_effect=boom):
            sb.compute_sweep_dataset(cfg, "l1")
        self.assertTrue(sb._sweep_skip_marker(cfg, "BOOM", method="l1").exists())

        # overwrite=True with a working fit: skip cleared, cache written.
        with mock.patch.object(sb.sh, "load_msigdb", return_value=library), \
             mock.patch.object(sb.ls, "sweep_subsample_betas", side_effect=real):
            sb.compute_sweep_dataset(cfg, "l1", overwrite=True)
        self.assertFalse(sb._sweep_skip_marker(cfg, "BOOM", method="l1").exists())
        self.assertIsNotNone(sb.load_sweep(cfg, "BOOM", method="l1"))

    def test_shrinking_k_recomputes_fresh(self):
        # Cached with k=2 subsamples; re-running with k=1 (mismatched subsample set) must
        # recompute fresh rather than union a stale {full,0,1} cache into a {full,0} request.
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        adata_path = self._write_adata(tmp, 26)
        library = {"PRESENT": ["A", "B", "C"]}
        cfg2 = self._cancer_cfg(tmp, adata_path, ["PRESENT"])      # sweep_k=2
        with mock.patch.object(sb.sh, "load_msigdb", return_value=library):
            sb.compute_sweep_dataset(cfg2, "l1")
        self.assertEqual(set(sb.load_sweep(cfg2, "PRESENT", method="l1")["subsample"].unique()),
                         {"full", 0, 1})

        cfg1 = self._cancer_cfg(tmp, adata_path, ["PRESENT"])
        cfg1.sweep_k = 1
        with mock.patch.object(sb.sh, "load_msigdb", return_value=library):
            sb.compute_sweep_dataset(cfg1, "l1")
        self.assertEqual(set(sb.load_sweep(cfg1, "PRESENT", method="l1")["subsample"].unique()),
                         {"full", 0})

    def test_colon_single_gene_is_swept(self):
        # The colon raw single gene (not a gene set) must be swept directly, like
        # `compute_run`'s raw branch -- parity with `compute_dataset`'s `all_targets`.
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        adata_path = self._write_adata(tmp, 27)
        cfg = _minimal_cfg(
            name="sweep_colon", kind="colon", adata_path=adata_path, cache_root=tmp,
            cell_type_col="cell_type", annot_col="cell_type", cell_type="T1",
            single_gene="A", t_cell_types=[], target_signatures=["PRESENT"], neg_controls=[],
            cancer_factors=[], vessel={}, method="l1", standardize=True,
            sweep_penalties={"l1": [0.1]}, sweep_k=2,
        )
        library = {"PRESENT": ["B", "C"]}
        with mock.patch.object(sb.sh, "load_msigdb", return_value=library):
            sb.compute_sweep_dataset(cfg, "l1")
        self.assertIsNotNone(sb.load_sweep(cfg, "A", method="l1"))       # raw gene swept
        self.assertIsNotNone(sb.load_sweep(cfg, "PRESENT", method="l1"))


if __name__ == "__main__":
    unittest.main()
