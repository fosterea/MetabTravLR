"""Tier-0 tests for the SLURM runner: `dataset_configs`, `run_spacetravlr`, `submit_spacetravlr`.

No training, no cluster, no real data -- every heavy call (`SpaceShip`, `_load_adata`,
`beta_analysis.*`, `Slurm.sbatch`) is patched out, so these assert the *orchestration*:
which stage runs, what gets deleted by `--overwrite`, and what sbatch would be handed.

The regression these guard hardest: SLURM opens its `--output` file before the job body
runs, so the log directory must exist at submit time and must NOT live inside the
`spacetravlr_output/` tree that a fresh setup job is about to create.
"""
import os
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest import mock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from metab_processing.metab_travlr_config import FOCUS_GENES
from metab_processing.SpaceTravLR import dataset_configs, run_spacetravlr, submit_spacetravlr
from metab_processing.SpaceTravLR.dataset_configs import DATASETS, dataset_paths, get_config

DATASET = "Primary_Dermal_Melanoma"


def write_tiny_h5ad(path, var_names=("A", "B")):
    """A real (if minimal) h5ad -- `setup_is_complete` now opens the file, so a text stub
    would read as a truncated/corrupt setup."""
    import anndata as ad
    import numpy as np

    adata = ad.AnnData(np.zeros((2, len(var_names)), dtype="float32"))
    adata.var_names = list(var_names)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path)


def make_dataset_tree(root, dataset=DATASET, setup=False, betadata_genes=()):
    """A skeleton dataset directory: the two required inputs, optionally a finished setup."""
    paths = dataset_paths(dataset, root)
    paths["dataset_dir"].mkdir(parents=True, exist_ok=True)
    paths["adata"].write_text("not really an h5ad")   # never opened; _load_adata is patched
    paths["selection_yaml"].parent.mkdir(parents=True, exist_ok=True)
    paths["selection_yaml"].write_text("metabolites: []\n")
    if setup:
        paths["input_data"].mkdir(parents=True, exist_ok=True)
        write_tiny_h5ad(paths["input_data"] / "_adata.h5ad")
        for name in ("celloracle_links.pkl", "tflinks.parquet"):
            (paths["input_data"] / name).write_text("x")
    for gene in betadata_genes:
        paths["betadata"].mkdir(parents=True, exist_ok=True)
        (paths["betadata"] / f"{gene}_betadata.parquet").write_text("x")
    return paths


class FakeObs:
    def __init__(self, columns):
        self.columns = list(columns)


class FakeAdata:
    """Only what run_dataset's artifacts stage touches."""

    def __init__(self, tiers=("Tier1", "Tier2", "Tier3")):
        self.obs = FakeObs(tiers)
        self.obsm = {}
        self.uns = {}
        self.written_to = None

    def write_h5ad(self, path):
        self.written_to = Path(path)
        self.written_to.write_text("fake h5ad")


# --------------------------------------------------------------------------- configs
class TestDatasetConfigs(unittest.TestCase):
    def test_known_datasets_resolve(self):
        for dataset in DATASETS:
            cfg = get_config(dataset)
            self.assertEqual(cfg["dataset"], dataset)
            self.assertIn("slurm", cfg)

    def test_unknown_dataset_raises(self):
        with self.assertRaises(KeyError):
            get_config("Not_A_Dataset")

    def test_all_datasets_share_the_same_focus_genes_by_default(self):
        # Foster's requirement: target genes are shared across datasets unless overridden.
        genes = {dataset: tuple(get_config(dataset)["focus_genes"]) for dataset in DATASETS}
        self.assertEqual(len(set(genes.values())), 1, genes)
        self.assertEqual(next(iter(genes.values())), tuple(FOCUS_GENES))

    def test_get_config_returns_a_deep_copy(self):
        cfg = get_config(DATASET)
        cfg["focus_genes"].append("BOGUS")
        cfg["slurm"]["partition"] = "nonsense"
        cfg["tiers"].append("Tier99")
        fresh = get_config(DATASET)
        self.assertNotIn("BOGUS", fresh["focus_genes"])
        self.assertNotIn("Tier99", fresh["tiers"])
        self.assertEqual(fresh["slurm"]["partition"], dataset_configs.DEFAULTS["slurm"]["partition"])
        self.assertNotIn("BOGUS", dataset_configs.DEFAULTS["focus_genes"])

    def test_dataset_override_merges_slurm_key_by_key(self):
        with mock.patch.dict(dataset_configs.DATASETS,
                             {"Tiny": {"slurm": {"time_hours": 2}, "cell_type_src": "leiden"}},
                             clear=False):
            cfg = get_config("Tiny")
        self.assertEqual(cfg["slurm"]["time_hours"], 2)
        self.assertEqual(cfg["cell_type_src"], "leiden")
        # untouched slurm keys still come from DEFAULTS
        self.assertEqual(cfg["slurm"]["partition"], dataset_configs.DEFAULTS["slurm"]["partition"])
        self.assertEqual(cfg["slurm"]["account"], dataset_configs.DEFAULTS["slurm"]["account"])

    def test_typo_in_dataset_override_raises(self):
        with mock.patch.dict(dataset_configs.DATASETS, {"Tiny": {"cell_type_scr": "oops"}}, clear=False):
            with self.assertRaises(KeyError):
                get_config("Tiny")


class TestDatasetPaths(unittest.TestCase):
    def test_layout_matches_quickstart(self):
        paths = dataset_paths(DATASET, "/data")
        self.assertEqual(paths["adata"], Path(f"/data/{DATASET}/adata.h5ad"))
        self.assertEqual(paths["outdir"], Path(f"/data/{DATASET}/spacetravlr_output"))
        self.assertEqual(paths["betadata"], paths["outdir"] / "betadata")
        self.assertEqual(paths["selection_yaml"],
                         Path(f"/data/{DATASET}/easy_download/harreman_outputs/metabolite_selection.yaml"))
        self.assertEqual(paths["metab_outdir"],
                         Path(f"/data/{DATASET}/easy_download/metabtravlr_outputs"))
        self.assertEqual(paths["beta_adata"], Path(f"/data/{DATASET}/spacetravlr_adata.h5ad"))

    def test_log_dir_is_outside_the_output_tree(self):
        # The whole reason this module exists: sbatch opens --output before the job runs,
        # and spacetravlr_output/ does not exist yet on a fresh dataset.
        paths = dataset_paths(DATASET, "/data")
        self.assertFalse(paths["log_dir"].is_relative_to(paths["outdir"]))
        self.assertFalse(paths["log_dir"].is_relative_to(paths["dataset_dir"]))
        self.assertEqual(paths["log_dir"].name, DATASET)
        self.assertEqual(paths["log_dir"].parent.name, "spacetravlr_logs")

    def test_log_root_is_a_sibling_of_harreman_logs(self):
        self.assertTrue(dataset_configs.LOG_ROOT.endswith("/spacetravlr_logs"))


# ------------------------------------------------------------------- setup detection
class TestSetupIsComplete(unittest.TestCase):
    def test_false_when_nothing_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(run_spacetravlr.setup_is_complete(Path(tmp) / "nope"))

    def test_true_only_when_every_artifact_is_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = make_dataset_tree(tmp, setup=True)
            self.assertTrue(run_spacetravlr.setup_is_complete(paths["outdir"]))
            for name in ("_adata.h5ad", "celloracle_links.pkl", "tflinks.parquet"):
                target = paths["input_data"] / name
                backup = target.read_bytes()
                target.unlink()
                self.assertFalse(run_spacetravlr.setup_is_complete(paths["outdir"]),
                                 f"should be incomplete without {name}")
                target.write_bytes(backup)
            self.assertTrue(run_spacetravlr.setup_is_complete(paths["outdir"]))

    def test_truncated_adata_counts_as_incomplete(self):
        # write_h5ad is not atomic: a job killed mid-write leaves a file that exists but
        # cannot be opened. Existence alone would make the resubmit skip setup and then
        # crash inside fit.
        with tempfile.TemporaryDirectory() as tmp:
            paths = make_dataset_tree(tmp, setup=True)
            target = paths["input_data"] / "_adata.h5ad"
            data = target.read_bytes()
            target.write_bytes(data[: len(data) // 2])
            self.assertFalse(run_spacetravlr.setup_is_complete(paths["outdir"]))

    def test_empty_adata_file_counts_as_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = make_dataset_tree(tmp, setup=True)
            (paths["input_data"] / "_adata.h5ad").write_bytes(b"")
            self.assertFalse(run_spacetravlr.setup_is_complete(paths["outdir"]))

    def test_does_not_depend_on_a_cwd_relative_launch_script(self):
        # SpaceShip.is_everything_ok() asserts os.path.isfile('launch.py'), which is never
        # true inside a SLURM job -- our check must not inherit that.
        with tempfile.TemporaryDirectory() as tmp:
            paths = make_dataset_tree(tmp, setup=True)
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                self.assertTrue(run_spacetravlr.setup_is_complete(paths["outdir"]))
            finally:
                os.chdir(cwd)


class TestIsolateCacheDir(unittest.TestCase):
    """genomepy opens a SQLite cache under `~/.cache` at import time; on Savio that is NFS,
    where SQLite locking fails outright (`OperationalError: locking protocol`).
    """

    def test_points_at_a_job_unique_local_path_and_creates_it(self):
        with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "12345"}, clear=False):
            os.environ.pop("XDG_CACHE_HOME", None)
            with mock.patch.object(run_spacetravlr.os, "makedirs") as makedirs:
                run_spacetravlr._isolate_cache_dir()
            cache = os.environ["XDG_CACHE_HOME"]
        self.assertEqual(cache, "/tmp/spacetravlr_cache_12345")
        makedirs.assert_called_once_with(cache, exist_ok=True)

    def test_two_jobs_get_different_cache_dirs(self):
        seen = []
        for job in ("111", "222"):
            with mock.patch.dict(os.environ, {"SLURM_JOB_ID": job}, clear=False):
                os.environ.pop("XDG_CACHE_HOME", None)
                with mock.patch.object(run_spacetravlr.os, "makedirs"):
                    run_spacetravlr._isolate_cache_dir()
                seen.append(os.environ["XDG_CACHE_HOME"])
        self.assertEqual(len(set(seen)), 2, seen)

    def test_is_never_the_nfs_home_cache(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("XDG_CACHE_HOME", None)
            os.environ.pop("SLURM_JOB_ID", None)
            with mock.patch.object(run_spacetravlr.os, "makedirs"):
                run_spacetravlr._isolate_cache_dir()
            cache = os.environ["XDG_CACHE_HOME"]
        self.assertTrue(cache.startswith("/tmp/"), cache)
        self.assertNotIn(str(Path.home()), cache)

    def test_an_explicit_setting_is_respected(self):
        with mock.patch.dict(os.environ, {"XDG_CACHE_HOME": "/somewhere/else"}, clear=False):
            with mock.patch.object(run_spacetravlr.os, "makedirs"):
                run_spacetravlr._isolate_cache_dir()
            self.assertEqual(os.environ["XDG_CACHE_HOME"], "/somewhere/else")

    def test_run_dataset_isolates_before_setup_imports_celloracle(self):
        # Ordering is the whole point: SpaceShip.setup_ -> run_celloracle_ -> import
        # celloracle_tmp -> genomepy opens the SQLite cache. Too late after that.
        order = []

        class OrderedShip:
            def __init__(self, name=None, outdir=None, genes=None):
                self.outdir = Path(outdir)

            def setup_(self, adata, overwrite=False, run_commot=False):
                order.append("setup_")
                (self.outdir / "input_data").mkdir(parents=True, exist_ok=True)
                write_tiny_h5ad(self.outdir / "input_data" / "_adata.h5ad")
                for name in ("celloracle_links.pkl", "tflinks.parquet"):
                    (self.outdir / "input_data" / name).write_text("x")

        with tempfile.TemporaryDirectory() as tmp:
            make_dataset_tree(tmp)
            with mock.patch.object(run_spacetravlr, "_isolate_cache_dir",
                                   side_effect=lambda: order.append("isolate")), \
                 mock.patch.object(run_spacetravlr, "SpaceShip", OrderedShip), \
                 mock.patch.object(run_spacetravlr, "_load_adata", lambda p, s: object()), \
                 mock.patch.object(run_spacetravlr, "_drop_tiny_clusters", lambda a, s: a):
                run_spacetravlr.run_dataset(DATASET, stages=["setup"], data_dir=tmp)

        self.assertEqual(order, ["isolate", "setup_"])


class TestJobIsActive(unittest.TestCase):
    """Decides whether a `.setup.lock` is genuinely held or was orphaned by a killed job.
    Every uncertain case must come back True, so we never steal a live lock."""

    def _squeue(self, stdout):
        return mock.patch.object(run_spacetravlr.subprocess, "run",
                                 return_value=mock.Mock(stdout=stdout))

    def test_job_still_in_the_queue_is_active(self):
        with self._squeue("36169480 savio3_gpu MetabTra fosteran R 0:28 1 n0264\n"):
            self.assertTrue(run_spacetravlr._job_is_active("36169480"))

    def test_job_absent_from_the_queue_is_not_active(self):
        with self._squeue(""):
            self.assertFalse(run_spacetravlr._job_is_active("36169480"))

    def test_whitespace_only_output_is_not_active(self):
        with self._squeue("\n  \n"):
            self.assertFalse(run_spacetravlr._job_is_active("36169480"))

    def test_a_hand_run_owner_is_assumed_active(self):
        self.assertTrue(run_spacetravlr._job_is_active("local-12345"))
        self.assertTrue(run_spacetravlr._job_is_active(""))

    def test_missing_squeue_is_assumed_active(self):
        with mock.patch.object(run_spacetravlr.subprocess, "run", side_effect=FileNotFoundError):
            self.assertTrue(run_spacetravlr._job_is_active("36169480"))

    def test_squeue_timeout_is_assumed_active(self):
        with mock.patch.object(run_spacetravlr.subprocess, "run",
                               side_effect=run_spacetravlr.subprocess.TimeoutExpired("squeue", 60)):
            self.assertTrue(run_spacetravlr._job_is_active("36169480"))


class TestProcessedVarNames(unittest.TestCase):
    def test_reads_var_names_from_a_real_h5ad_without_loading_it(self):
        import anndata as ad
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            paths = dataset_paths(DATASET, tmp)
            paths["input_data"].mkdir(parents=True)
            adata = ad.AnnData(np.zeros((3, 4), dtype="float32"))
            adata.var_names = ["CD4", "CD3E", "ABCA1", "ATP7A"]
            adata.write_h5ad(paths["input_data"] / "_adata.h5ad")
            self.assertEqual(run_spacetravlr._processed_var_names(paths),
                             ["CD4", "CD3E", "ABCA1", "ATP7A"])


# ----------------------------------------------------------------------- run_dataset
# ------------------------------------------------- MAGIC tiny-cluster drop at load
def clustered_adata(**columns):
    """A real AnnData whose only meaningful content is `.obs` cluster columns.

    Each kwarg is `column=[label per cell]`, stored categorical the way scanpy's leiden
    output is -- that dtype is what makes the unused-category case possible. `X` is a
    zero matrix just so subsetting (`_drop_tiny_clusters` returns `adata[keep].copy()`)
    behaves like the real thing.
    """
    import anndata as ad
    import numpy as np
    import pandas as pd

    n = len(next(iter(columns.values())))
    obs = pd.DataFrame({k: pd.Series(v, dtype="category") for k, v in columns.items()})
    adata = ad.AnnData(np.zeros((n, 1), dtype="float32"), obs=obs)
    return adata


class TestDropTinyClusters(unittest.TestCase):
    """Human_Breast died an hour into setup on a sub-16-cell cluster
    (`ValueError: n_neighbors = 16, n_samples_fit = 10`). Drop those outlier cells at
    load instead -- and, if too many cells are in tiny clusters, stop and ask for a
    coarser resolution rather than silently discard a meaningful slice."""

    def test_keeps_every_cell_when_all_clusters_big_enough(self):
        adata = clustered_adata(res=["a"] * 20 + ["b"] * 16)
        out = run_spacetravlr._drop_tiny_clusters(adata, "res")
        self.assertEqual(out.n_obs, 36)

    def test_boundary_cluster_of_exactly_the_minimum_is_kept(self):
        adata = clustered_adata(
            res=["a"] * 20 + ["b"] * run_spacetravlr.MAGIC_MIN_CLUSTER_CELLS)
        out = run_spacetravlr._drop_tiny_clusters(adata, "res")
        self.assertEqual(out.n_obs, 20 + run_spacetravlr.MAGIC_MIN_CLUSTER_CELLS)

    def test_drops_cells_in_a_sub_minimum_cluster(self):
        adata = clustered_adata(res=["big"] * 1000 + ["tiny"] * 10)
        out = run_spacetravlr._drop_tiny_clusters(adata, "res")
        self.assertEqual(out.n_obs, 1000)
        # The emptied category is gone, so impute_clusterwise never visits it.
        self.assertNotIn("tiny", list(out.obs["res"].cat.categories))
        self.assertEqual(set(out.obs["res"]), {"big"})

    def test_source_adata_is_not_mutated(self):
        adata = clustered_adata(res=["big"] * 1000 + ["tiny"] * 10)
        run_spacetravlr._drop_tiny_clusters(adata, "res")
        # The drop trims a copy; the object we were handed still has all its cells.
        self.assertEqual(adata.n_obs, 1010)
        self.assertIn("tiny", set(adata.obs["res"]))

    def test_drop_is_logged_with_count_and_cluster(self):
        adata = clustered_adata(res=["big"] * 1000 + ["tiny"] * 10)
        msgs = []
        with mock.patch.object(run_spacetravlr, "_log", lambda m: msgs.append(m)):
            run_spacetravlr._drop_tiny_clusters(adata, "res")
        blob = "\n".join(msgs)
        self.assertIn("dropping", blob)
        self.assertIn("tiny", blob)
        self.assertIn("10", blob)

    def test_too_many_tiny_cells_raise_with_a_resolution_table(self):
        # 10 of 210 cells (~4.8%) exceeds the drop fraction -- the resolution is too fine,
        # so one failed run should tell Foster which sibling to switch to.
        adata = clustered_adata(**{
            "leiden_scVI_res_0.5": ["a"] * 200 + ["b"] * 10,
            "leiden_scVI_res_0.25": ["a"] * 210,
            "leiden_scVI_res_1": ["a"] * 195 + ["b"] * 10 + ["c"] * 5,
        })
        with self.assertRaises(ValueError) as ctx:
            run_spacetravlr._drop_tiny_clusters(adata, "leiden_scVI_res_0.5")
        msg = str(ctx.exception)
        coarser = [ln for ln in msg.splitlines() if "leiden_scVI_res_0.25" in ln]
        finer = [ln for ln in msg.splitlines() if "leiden_scVI_res_1" in ln]
        self.assertTrue(coarser and coarser[0].strip().endswith("yes"), msg)
        self.assertTrue(finer and finer[0].strip().endswith("no"), msg)

    def test_unused_categories_are_not_counted_as_empty_clusters(self):
        # A category with no cells is never visited by impute_clusterwise's
        # `obs[annot].unique()`, so it must not trip the drop.
        import anndata as ad
        import numpy as np
        import pandas as pd

        col = pd.Series(["a"] * 20 + ["b"] * 20,
                        dtype=pd.CategoricalDtype(["a", "b", "ghost"]))
        adata = ad.AnnData(np.zeros((40, 1), dtype="float32"),
                           obs=pd.DataFrame({"res": col}))
        out = run_spacetravlr._drop_tiny_clusters(adata, "res")
        self.assertEqual(out.n_obs, 40)

    def test_sibling_annotations_are_coarsest_first_and_exclude_self(self):
        adata = clustered_adata(**{
            "leiden_scVI_res_0.5": ["a"] * 20,
            "leiden_scVI_res_2": ["a"] * 20,
            "leiden_scVI_res_0.25": ["a"] * 20,
            "some_other_column": ["a"] * 20,
        })
        siblings = run_spacetravlr._sibling_annotations(adata.obs, "leiden_scVI_res_0.5")
        self.assertEqual(siblings, ["leiden_scVI_res_0.25", "leiden_scVI_res_2"])

    def test_no_siblings_when_the_column_has_no_numeric_suffix(self):
        adata = clustered_adata(cell_type=["a"] * 20, other=["a"] * 20)
        self.assertEqual(run_spacetravlr._sibling_annotations(adata.obs, "cell_type"), [])


class TestDatasetResolutions(unittest.TestCase):
    def test_every_dataset_keeps_the_default_resolution(self):
        # Tiny clusters are handled by dropping outliers at load, not by per-dataset
        # resolution overrides -- so no dataset (Human_Breast included) overrides it.
        for dataset in DATASETS:
            self.assertEqual(get_config(dataset)["cell_type_src"],
                             dataset_configs.DEFAULTS["cell_type_src"], dataset)


class RunDatasetCase(unittest.TestCase):
    """Shared harness: patch out everything heavy, record what got called."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        self.adata = FakeAdata()
        self.setup_calls = []

        outer = self

        class FakeShip:
            def __init__(self, name=None, outdir=None, genes=None):
                self.outdir = Path(outdir)
                self.genes = genes

            def setup_(self, adata, overwrite=False, run_commot=False):
                outer.setup_calls.append({"overwrite": overwrite, "run_commot": run_commot})
                (self.outdir / "input_data").mkdir(parents=True, exist_ok=True)
                write_tiny_h5ad(self.outdir / "input_data" / "_adata.h5ad")
                for name in ("celloracle_links.pkl", "tflinks.parquet"):
                    (self.outdir / "input_data" / name).write_text("x")

            def fit(self, metabolites=None, **kwargs):
                betadata = self.outdir / "betadata"
                before = sorted(p.name for p in betadata.glob("*_betadata.parquet")) \
                    if betadata.exists() else []
                outer.fit_calls.append(
                    {"metabolites": metabolites, "kwargs": kwargs, "betadata_before": before})
                betadata.mkdir(parents=True, exist_ok=True)
                for gene in self.genes:
                    (betadata / f"{gene}_betadata.parquet").write_text("x")

        self.fit_calls = []
        self.artifact_calls = []

        def record(name):
            def _fn(*a, **kw):
                self.artifact_calls.append((name, a, kw))
                if name == "betas_to_adata":
                    adata = a[0]
                    adata.uns["beta_modulators"] = {"CD4": ["metab@ABCA1"]}
                    adata.obsm["beta_CD4"] = _Shape((6, 1))
                return {}
            return _fn

        patches = [
            mock.patch.object(run_spacetravlr, "SpaceShip", FakeShip),
            mock.patch.object(run_spacetravlr, "_load_adata", lambda paths, src: self.adata),
            # FakeAdata has no real obs frame; the drop has its own tests below. It is a
            # pass-through here, returning the adata unchanged the way "nothing to drop" does.
            mock.patch.object(run_spacetravlr, "_drop_tiny_clusters", lambda adata, src: adata),
            mock.patch.object(run_spacetravlr, "load_metabolites",
                              lambda path, var_names=None: ({"m1": [("A", "B")]}, {"m1": [("A", "B")]})),
            mock.patch.object(run_spacetravlr, "_processed_var_names", lambda paths: ["A", "B"]),
            mock.patch.object(run_spacetravlr.beta_analysis, "write_metabolites", record("write_metabolites")),
            mock.patch.object(run_spacetravlr.beta_analysis, "write_histograms", record("write_histograms")),
            mock.patch.object(run_spacetravlr.beta_analysis, "betas_to_adata", record("betas_to_adata")),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def run_it(self, **kwargs):
        kwargs.setdefault("data_dir", self.root)
        return run_spacetravlr.run_dataset(DATASET, **kwargs)


class _Shape:
    """Stand-in for a numpy array -- run_dataset only logs `.shape`."""

    def __init__(self, shape):
        self.shape = shape


class TestRunDatasetSetupStage(RunDatasetCase):
    def test_missing_input_raises_before_doing_any_work(self):
        with self.assertRaises(FileNotFoundError):
            self.run_it(stages=["setup"])
        self.assertEqual(self.setup_calls, [])

    def test_setup_runs_when_absent(self):
        make_dataset_tree(self.root)
        self.run_it(stages=["setup"])
        self.assertEqual(len(self.setup_calls), 1)
        # overwrite=True is always passed: it only bypasses setup_'s "dir exists" guard,
        # it deletes nothing. We own the directory lifecycle.
        self.assertTrue(self.setup_calls[0]["overwrite"])
        self.assertFalse(self.setup_calls[0]["run_commot"])

    def test_bad_cluster_sizes_abort_before_setup_and_release_the_lock(self):
        make_dataset_tree(self.root)
        paths = dataset_paths(DATASET, self.root)
        boom = ValueError("resolution too fine")

        def raiser(adata, src):
            raise boom

        with mock.patch.object(run_spacetravlr, "_drop_tiny_clusters", raiser):
            with self.assertRaises(ValueError):
                self.run_it(stages=["setup"])
        # The expensive part never started, and a retry is not blocked by a stale lock.
        self.assertEqual(self.setup_calls, [])
        self.assertFalse((paths["outdir"] / ".setup.lock").exists())

    def test_setup_skipped_when_already_complete(self):
        make_dataset_tree(self.root, setup=True)
        self.run_it(stages=["setup"])
        self.assertEqual(self.setup_calls, [])

    def test_overwrite_redoes_setup_and_keeps_betadata(self):
        # Foster's chosen semantics: --overwrite clears input_data only.
        paths = make_dataset_tree(self.root, setup=True, betadata_genes=["CD4", "CD3E"])
        marker = paths["input_data"] / "stale.txt"
        marker.write_text("stale")
        self.run_it(stages=["setup"], overwrite=True)
        self.assertEqual(len(self.setup_calls), 1)
        self.assertFalse(marker.exists(), "input_data should have been cleared")
        self.assertEqual(
            sorted(p.name for p in paths["betadata"].glob("*_betadata.parquet")),
            ["CD3E_betadata.parquet", "CD4_betadata.parquet"])

    def test_clear_betadata_removes_trained_genes(self):
        paths = make_dataset_tree(self.root, setup=True, betadata_genes=["CD4"])
        self.run_it(stages=["setup"], overwrite=True, clear_betadata=True)
        self.assertEqual(list(paths["betadata"].glob("*_betadata.parquet")), [])

    def test_clear_betadata_without_overwrite_keeps_setup(self):
        paths = make_dataset_tree(self.root, setup=True, betadata_genes=["CD4"])
        self.run_it(stages=["setup"], clear_betadata=True)
        self.assertEqual(self.setup_calls, [], "setup was complete; should not rerun")
        self.assertFalse(paths["betadata"].exists())

    def test_overwrite_without_the_setup_stage_is_rejected(self):
        # Silently ignoring it would be worse: --stage fit --overwrite would look like a
        # clean rerun while reusing the old preprocessing.
        make_dataset_tree(self.root, setup=True, betadata_genes=["CD4"])
        with self.assertRaises(ValueError):
            self.run_it(stages=["fit"], overwrite=True)

    def _write_lock(self, owner):
        paths = dataset_paths(DATASET, self.root)
        paths["outdir"].mkdir(parents=True, exist_ok=True)
        lock = paths["outdir"] / ".setup.lock"
        lock.write_text(f"{owner}\nstarted whenever\n")
        return lock

    def test_setup_lock_blocks_a_setup_whose_job_is_still_active(self):
        make_dataset_tree(self.root)
        self._write_lock("999")
        with mock.patch.object(run_spacetravlr, "_job_is_active", return_value=True):
            with self.assertRaises(RuntimeError):
                self.run_it(stages=["setup"])
        self.assertEqual(self.setup_calls, [])

    def test_a_lock_orphaned_by_a_dead_job_is_taken_over(self):
        # OOM kill and scancel are SIGKILL, so the `finally` never runs. Without this,
        # one OOM blocks every resubmit until the file is deleted by hand.
        make_dataset_tree(self.root)
        lock = self._write_lock("36169480")
        with mock.patch.object(run_spacetravlr, "_job_is_active", return_value=False):
            self.run_it(stages=["setup"])
        self.assertEqual(len(self.setup_calls), 1)
        self.assertFalse(lock.exists())

    def test_the_lock_records_the_slurm_job_id(self):
        make_dataset_tree(self.root)
        paths = dataset_paths(DATASET, self.root)
        seen = {}

        def peek(*a, **kw):
            # read the lock from inside the critical section
            seen["owner"] = (paths["outdir"] / ".setup.lock").read_text().splitlines()[0]
            return object()

        with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "424242"}, clear=False), \
             mock.patch.object(run_spacetravlr, "_load_adata", side_effect=peek):
            self.run_it(stages=["setup"])
        self.assertEqual(seen["owner"], "424242",
                         "the job id is what lets a later run test liveness")

    def test_setup_lock_is_released_afterwards(self):
        paths = make_dataset_tree(self.root)
        self.run_it(stages=["setup"])
        self.assertFalse((paths["outdir"] / ".setup.lock").exists())

    def test_setup_lock_is_released_when_setup_raises(self):
        make_dataset_tree(self.root)
        paths = dataset_paths(DATASET, self.root)
        with mock.patch.object(run_spacetravlr, "_load_adata", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                self.run_it(stages=["setup"])
        self.assertFalse((paths["outdir"] / ".setup.lock").exists(),
                         "a crashed setup must not leave a lock that blocks every resubmit")


class TestRunDatasetFitStage(RunDatasetCase):
    def test_fit_refuses_without_setup(self):
        make_dataset_tree(self.root)
        with self.assertRaises(RuntimeError):
            self.run_it(stages=["fit"])

    def test_clear_betadata_applies_to_a_fit_only_run(self):
        # The deletion used to live inside the setup branch, so `--stage fit
        # --clear-betadata` silently retrained nothing (the queue skips genes whose
        # parquet exists).
        make_dataset_tree(self.root, setup=True, betadata_genes=["CD4"])
        self.run_it(stages=["fit"], clear_betadata=True)
        self.assertEqual(self.fit_calls[0]["betadata_before"], [],
                         "betadata should already be gone when fit starts")

    def test_fit_only_run_without_the_flag_keeps_trained_genes(self):
        make_dataset_tree(self.root, setup=True, betadata_genes=["CD4"])
        self.run_it(stages=["fit"])
        self.assertEqual(self.fit_calls[0]["betadata_before"], ["CD4_betadata.parquet"],
                         "a plain resubmit must resume, not retrain")

    def test_fit_passes_metabolites_and_config_kwargs(self):
        make_dataset_tree(self.root, setup=True)
        with mock.patch.dict(dataset_configs.DATASETS,
                             {DATASET: {"fit_kwargs": {"max_epochs": 3}}}, clear=False):
            self.run_it(stages=["fit"])
        self.assertEqual(len(self.fit_calls), 1)
        self.assertEqual(self.fit_calls[0]["metabolites"], {"m1": [("A", "B")]})
        self.assertEqual(self.fit_calls[0]["kwargs"], {"max_epochs": 3})


class TestRunDatasetArtifactsStage(RunDatasetCase):
    def test_artifacts_refuse_without_betadata(self):
        make_dataset_tree(self.root, setup=True)
        with self.assertRaises(RuntimeError):
            self.run_it(stages=["artifacts"])

    def test_artifacts_refuse_when_betadata_is_all_stale_genes(self):
        # An artifacts-only rerun after focus_genes changed: parquets exist, but none of
        # them is a current target gene. Without this guard beta_analysis happily writes
        # header-only CSVs and an empty beta_modulators, and the job exits 0.
        make_dataset_tree(self.root, setup=True, betadata_genes=["OLD_GENE_A", "OLD_GENE_B"])
        with self.assertRaises(RuntimeError) as ctx:
            self.run_it(stages=["artifacts"])
        self.assertIn("OLD_GENE_A", str(ctx.exception))

    def test_artifacts_proceed_on_a_partial_gene_set(self):
        # Genes orphan legitimately (no regulators, or all-zero betas), so a subset is fine.
        make_dataset_tree(self.root, setup=True, betadata_genes=["CD4"])
        self.run_it(stages=["artifacts"])
        self.assertEqual([n for n, _, _ in self.artifact_calls],
                         ["write_metabolites", "write_histograms", "betas_to_adata"])

    def test_artifacts_write_everything_and_the_beta_adata(self):
        paths = make_dataset_tree(self.root, setup=True, betadata_genes=["CD4"])
        self.run_it(stages=["artifacts"])
        names = [name for name, _, _ in self.artifact_calls]
        self.assertEqual(names, ["write_metabolites", "write_histograms", "betas_to_adata"])
        self.assertEqual(self.adata.written_to, paths["beta_adata"])
        self.assertTrue(paths["beta_adata"].exists())

    def test_artifacts_use_configured_tiers_present_in_obs(self):
        make_dataset_tree(self.root, setup=True, betadata_genes=["CD4"])
        self.adata.obs = FakeObs(["Tier1", "Tier3", "other"])
        self.run_it(stages=["artifacts"])
        _, args, _ = self.artifact_calls[0]
        self.assertEqual(args[2], ["Tier1", "Tier3"])

    def test_artifacts_raise_when_no_tier_is_present(self):
        make_dataset_tree(self.root, setup=True, betadata_genes=["CD4"])
        self.adata.obs = FakeObs(["leiden"])
        with self.assertRaises(ValueError):
            self.run_it(stages=["artifacts"])

    def test_histograms_are_plotted(self):
        make_dataset_tree(self.root, setup=True, betadata_genes=["CD4"])
        self.run_it(stages=["artifacts"])
        _, _, kwargs = self.artifact_calls[1]
        self.assertTrue(kwargs["plot"], "histograms.png is one of the artifacts we want")

    def test_beta_group_default_keeps_every_modulator_group(self):
        make_dataset_tree(self.root, setup=True, betadata_genes=["CD4"])
        self.run_it(stages=["artifacts"])
        _, _, kwargs = self.artifact_calls[2]
        self.assertIsNone(kwargs["group"], "default is all groups (tf+lr+ltf+metab)")

    def test_beta_group_is_configurable_per_dataset(self):
        make_dataset_tree(self.root, setup=True, betadata_genes=["CD4"])
        with mock.patch.dict(dataset_configs.DATASETS, {DATASET: {"beta_group": "metab"}}, clear=False):
            self.run_it(stages=["artifacts"])
        _, _, kwargs = self.artifact_calls[2]
        self.assertEqual(kwargs["group"], "metab")


class TestRunDatasetAllStages(RunDatasetCase):
    def test_full_run_from_scratch(self):
        paths = make_dataset_tree(self.root)
        self.run_it()
        self.assertEqual(len(self.setup_calls), 1)
        self.assertEqual(len(self.fit_calls), 1)
        self.assertEqual([n for n, _, _ in self.artifact_calls],
                         ["write_metabolites", "write_histograms", "betas_to_adata"])
        self.assertTrue(paths["beta_adata"].exists())


class TestArtifactsAgainstRealBetaAnalysis(unittest.TestCase):
    """The stage tests above mock `beta_analysis` and `SpaceShip` out, so they cannot catch
    signature drift between the runner and what it calls. This one mocks NOTHING: real
    parquets, a real AnnData on disk, the real `beta_analysis` functions, real h5ad write.
    """

    def test_artifacts_stage_end_to_end(self):
        import anndata as ad
        import numpy as np
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmp:
            paths = dataset_paths(DATASET, tmp)
            cfg = get_config(DATASET)
            cells = [f"c{i}" for i in range(6)]

            # A real raw adata with the configured cell-type column, so _load_adata is real too.
            raw = ad.AnnData(np.ones((6, 2), dtype="float32"))
            raw.obs_names = cells
            raw.var_names = ["ABCA1", "ATP7A"]
            raw.obs[cfg["cell_type_src"]] = ["0"] * 3 + ["1"] * 3
            raw.obs["Tier1"] = ["T Cell"] * 3 + ["Tumor"] * 3
            paths["dataset_dir"].mkdir(parents=True, exist_ok=True)
            raw.write_h5ad(paths["adata"])
            paths["selection_yaml"].parent.mkdir(parents=True, exist_ok=True)
            paths["selection_yaml"].write_text("metabolites: []\n")

            # Two trained target genes, two metabolites (one merged) + one TF modulator each.
            betas = pd.DataFrame(
                {
                    "beta_metab@Glucose": np.arange(6, dtype="float32"),
                    "beta_metab@Copper|Zinc": np.linspace(-1, 1, 6, dtype="float32"),
                    "beta_STAT1": np.full(6, 0.5, dtype="float32"),
                },
                index=cells,
            )
            paths["betadata"].mkdir(parents=True, exist_ok=True)
            for gene in ("CD4", "CD3E"):
                betas.to_parquet(paths["betadata"] / f"{gene}_betadata.parquet")

            run_spacetravlr.run_dataset(DATASET, stages=["artifacts"], data_dir=tmp)

            tier_dir = paths["metab_outdir"] / "Tier1"
            metabolites = pd.read_csv(tier_dir / "metabolites.csv")
            self.assertEqual(set(metabolites["gene"]), {"CD4", "CD3E"})
            self.assertEqual(set(metabolites["metabolite"]), {"Glucose", "Copper|Zinc"},
                             "metabolites.csv is the metab group only, 'metab@' stripped")
            self.assertEqual(set(metabolites["cell_type"]), {"T Cell", "Tumor"})

            hist = pd.read_csv(tier_dir / "histograms.csv")
            self.assertEqual(set(hist["group"]), {"metab", "tf"})
            self.assertTrue((tier_dir / "histograms.png").is_file())

            # Only Tier1 is in obs; Tier2/Tier3 must be skipped, not crash.
            self.assertFalse((paths["metab_outdir"] / "Tier2").exists())

            out = ad.read_h5ad(paths["beta_adata"])
            self.assertEqual(out.obsm["beta_CD4"].shape, (6, 3),
                             "group=None keeps every modulator group")
            self.assertEqual(list(out.uns["beta_modulators"]["CD4"]),
                             ["metab@Glucose", "metab@Copper|Zinc", "STAT1"])
            np.testing.assert_allclose(out.obsm["beta_CD4"][:, 0], np.arange(6))

    def test_artifacts_stage_end_to_end_with_metab_only_beta_group(self):
        import anndata as ad
        import numpy as np
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmp:
            paths = dataset_paths(DATASET, tmp)
            cfg = get_config(DATASET)
            cells = [f"c{i}" for i in range(4)]
            raw = ad.AnnData(np.ones((4, 1), dtype="float32"))
            raw.obs_names = cells
            raw.obs[cfg["cell_type_src"]] = ["0"] * 4
            raw.obs["Tier1"] = ["T Cell"] * 4
            paths["dataset_dir"].mkdir(parents=True, exist_ok=True)
            raw.write_h5ad(paths["adata"])
            paths["selection_yaml"].parent.mkdir(parents=True, exist_ok=True)
            paths["selection_yaml"].write_text("metabolites: []\n")
            paths["betadata"].mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"beta_ABCA1@ABCA1": np.ones(4, dtype="float32"),
                          "beta_STAT1": np.ones(4, dtype="float32")},
                         index=cells).to_parquet(paths["betadata"] / "CD4_betadata.parquet")

            with mock.patch.dict(dataset_configs.DATASETS, {DATASET: {"beta_group": "metab"}},
                                 clear=False):
                run_spacetravlr.run_dataset(DATASET, stages=["artifacts"], data_dir=tmp)

            out = ad.read_h5ad(paths["beta_adata"])
            self.assertEqual(out.obsm["beta_CD4"].shape, (4, 1))
            self.assertEqual(list(out.uns["beta_modulators"]["CD4"]), ["ABCA1@ABCA1"])

    def test_artifacts_stage_writes_x_metab_for_real_metabolites(self):
        """End-to-end x-half wiring: a real metabolite selection + processed `_adata.h5ad`
        (imputed_count + spatial) + `run_params.json` make the artifacts stage attach
        `x_metab` (the communication scores) alongside the betas, matching a direct
        `compute_metab_x` on the processed adata. Guards the runner<->beta_analysis signature
        and the compute-once/reindex path. Needs torch (real diffusion)."""
        import anndata as ad
        import numpy as np
        import pandas as pd

        from metab_processing.SpaceTravLR import beta_analysis

        with tempfile.TemporaryDirectory() as tmp:
            paths = dataset_paths(DATASET, tmp)
            cfg = get_config(DATASET)
            cells = [f"c{i}" for i in range(6)]
            genes = ["ABCA1", "ATP7A"]  # export, import transporters

            rng = np.random.default_rng(0)
            # Raw display adata (the file _load_adata reads) -- no imputed_count needed here.
            raw = ad.AnnData(rng.random((6, 2)).astype("float32"))
            raw.obs_names = cells
            raw.var_names = genes
            raw.obs[cfg["cell_type_src"]] = ["0"] * 3 + ["1"] * 3
            raw.obs["Tier1"] = ["T Cell"] * 3 + ["Tumor"] * 3
            paths["dataset_dir"].mkdir(parents=True, exist_ok=True)
            raw.write_h5ad(paths["adata"])

            # Processed adata the diffusion reads: imputed_count layer + spatial + same cells.
            proc = raw.copy()
            proc.layers["imputed_count"] = rng.random((6, 2)).astype("float32")
            proc.obsm["spatial"] = rng.uniform(0, 400, size=(6, 2))
            paths["input_data"].mkdir(parents=True, exist_ok=True)
            proc.write_h5ad(paths["input_data"] / "_adata.h5ad")

            paths["selection_yaml"].parent.mkdir(parents=True, exist_ok=True)
            paths["selection_yaml"].write_text(
                "metabolites:\n- name: Copper\n  gene_pairs:\n  - [ABCA1, ATP7A]\n")

            paths["betadata"].mkdir(parents=True, exist_ok=True)
            # metab@Copper is orientation-expanded to two pairs -> one summed column.
            pd.DataFrame({"beta_metab@Copper": np.arange(6, dtype="float32"),
                          "beta_STAT1": np.full(6, 0.5, dtype="float32")},
                         index=cells).to_parquet(paths["betadata"] / "CD4_betadata.parquet")
            (paths["betadata"] / "run_params.json").write_text(
                '{"radius": 100, "contact_distance": 30, "scale_factor": 100, '
                '"layer": "imputed_count"}')

            run_spacetravlr.run_dataset(DATASET, stages=["artifacts"], data_dir=tmp)

            out = ad.read_h5ad(paths["beta_adata"])
            self.assertIn("x_metab", out.obsm)
            self.assertEqual(list(out.uns["x_metab_modulators"]), ["metab@Copper"])
            self.assertEqual(out.obsm["x_metab"].shape, (6, 1))

            from metab_processing.SpaceTravLR.metab_loader import load_metabolites
            metabolites, _ = load_metabolites(paths["selection_yaml"], var_names=genes)
            expected = beta_analysis.compute_metab_x(
                proc.copy(), metabolites, radius=100, contact_distance=30,
                scale_factor=100, layer="imputed_count")
            np.testing.assert_allclose(
                out.obsm["x_metab"][:, 0], expected["metab@Copper"].values, rtol=1e-6, atol=1e-6)


class TestRunSpacetravlrCli(unittest.TestCase):
    def test_stages_are_canonically_ordered_regardless_of_flag_order(self):
        with mock.patch.object(run_spacetravlr, "run_dataset") as fake:
            run_spacetravlr.main(["--dataset", DATASET, "--stage", "artifacts", "setup"])
        self.assertEqual(fake.call_args.kwargs["stages"], ["setup", "artifacts"])

    def test_defaults_run_every_stage(self):
        with mock.patch.object(run_spacetravlr, "run_dataset") as fake:
            run_spacetravlr.main(["--dataset", DATASET])
        self.assertEqual(fake.call_args.kwargs["stages"], ["setup", "fit", "artifacts"])
        self.assertFalse(fake.call_args.kwargs["overwrite"])
        self.assertFalse(fake.call_args.kwargs["clear_betadata"])

    def test_flags_are_forwarded(self):
        with mock.patch.object(run_spacetravlr, "run_dataset") as fake:
            run_spacetravlr.main(["--dataset", DATASET, "--overwrite", "--clear-betadata"])
        self.assertTrue(fake.call_args.kwargs["overwrite"])
        self.assertTrue(fake.call_args.kwargs["clear_betadata"])

    def test_unknown_dataset_is_rejected_by_argparse(self):
        with self.assertRaises(SystemExit):
            run_spacetravlr.main(["--dataset", "Not_A_Dataset"])

    def test_unknown_stage_is_rejected_by_argparse(self):
        with self.assertRaises(SystemExit):
            run_spacetravlr.main(["--dataset", DATASET, "--stage", "train"])


# --------------------------------------------------------------------------- submit
class TestBuildCommand(unittest.TestCase):
    def test_minimal_command(self):
        cmd = build = submit_spacetravlr.build_command(DATASET, python_path="/env/bin/python")
        self.assertTrue(cmd.startswith("/env/bin/python "))
        self.assertIn("run_spacetravlr.py", cmd)
        self.assertIn(f"--dataset {DATASET}", cmd)
        self.assertNotIn("--stage", cmd)
        self.assertNotIn("--overwrite", cmd)
        self.assertNotIn("--data-dir", build)

    def test_all_flags(self):
        cmd = submit_spacetravlr.build_command(
            DATASET, stages=["setup", "fit"], overwrite=True, clear_betadata=True,
            data_dir="/other/data", python_path="python")
        self.assertIn("--stage setup fit", cmd)
        self.assertIn("--overwrite", cmd)
        self.assertIn("--clear-betadata", cmd)
        self.assertIn("--data-dir /other/data", cmd)

    def test_run_script_exists(self):
        # A rename would otherwise fail only once the job is already queued on Savio.
        self.assertTrue(submit_spacetravlr.RUN_SCRIPT.is_file(), submit_spacetravlr.RUN_SCRIPT)


class TestSubmit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p = mock.patch.object(dataset_configs, "LOG_ROOT", os.path.join(self.tmp.name, "spacetravlr_logs"))
        p.start()
        self.addCleanup(p.stop)

    def test_dry_run_submits_nothing_and_has_no_side_effects(self):
        # No mkdir either -- a dry run must be previewable from a laptop, where the Savio
        # log root is not writable.
        with mock.patch("simple_slurm.Slurm") as slurm:
            job = submit_spacetravlr.submit(DATASET, dry_run=True)
        self.assertIsNone(job)
        slurm.assert_not_called()
        self.assertFalse((Path(dataset_configs.LOG_ROOT) / DATASET).exists())

    def test_real_submit_creates_the_log_dir_first(self):
        # The log dir MUST exist before sbatch or the job dies opening its output file.
        created = {}

        def check(*a, **kw):
            created["existed_at_sbatch"] = Path(kw["output"]).parent.is_dir()
            return mock.DEFAULT

        with mock.patch("simple_slurm.Slurm", side_effect=check) as slurm:
            slurm.return_value.sbatch.return_value = 1
            submit_spacetravlr.submit(DATASET)
        self.assertTrue(created["existed_at_sbatch"])

    def _capture(self, **kwargs):
        with mock.patch("simple_slurm.Slurm") as slurm:
            slurm.return_value.sbatch.return_value = 4242
            job = submit_spacetravlr.submit(DATASET, **kwargs)
        return job, slurm.call_args.kwargs, slurm.return_value.sbatch.call_args.args[0]

    def test_sbatch_settings_come_from_the_dataset_config(self):
        job, sb, cmd = self._capture()
        cfg = get_config(DATASET)["slurm"]
        self.assertEqual(job, 4242)
        self.assertEqual(sb["account"], cfg["account"])
        self.assertEqual(sb["partition"], cfg["partition"])
        self.assertEqual(sb["qos"], cfg["qos"])
        self.assertEqual(sb["gres"], cfg["gres"])
        self.assertEqual(sb["cpus_per_task"], cfg["cpus_per_task"])
        self.assertEqual(sb["time"], timedelta(hours=cfg["time_hours"]))
        self.assertTrue(sb["ignore_pbs"])
        self.assertEqual(sb["job_name"], f'{cfg["job_name"]}_{DATASET}')
        self.assertIn(f"--dataset {DATASET}", cmd)
        self.assertTrue(cmd.startswith(cfg["python_path"]))

    def test_log_file_lands_under_the_dataset_log_dir(self):
        _, sb, _ = self._capture()
        outlog = Path(sb["output"])
        self.assertEqual(outlog.parent, Path(dataset_configs.LOG_ROOT) / DATASET)
        self.assertTrue(outlog.parent.is_dir())
        self.assertTrue(outlog.name.startswith("all_"))
        self.assertTrue(outlog.name.endswith(".log"))

    def test_log_file_name_records_the_stages(self):
        _, sb, _ = self._capture(stages=["artifacts"])
        self.assertTrue(Path(sb["output"]).name.startswith("artifacts_"))

    def test_slurm_overrides_apply(self):
        _, sb, _ = self._capture(time_hours=24, partition="savio3", cpus_per_task=2)
        self.assertEqual(sb["time"], timedelta(hours=24))
        self.assertEqual(sb["partition"], "savio3")
        self.assertEqual(sb["cpus_per_task"], 2)
        # unrelated keys still come from the config
        self.assertEqual(sb["qos"], get_config(DATASET)["slurm"]["qos"])

    def test_overrides_do_not_leak_into_later_submissions(self):
        self._capture(time_hours=99)
        _, sb, _ = self._capture()
        self.assertEqual(sb["time"], timedelta(hours=get_config(DATASET)["slurm"]["time_hours"]))

    def test_gres_none_drops_the_gpu_request(self):
        _, sb, _ = self._capture(gres=None)
        self.assertNotIn("gres", sb)

    def test_unknown_slurm_override_raises(self):
        with self.assertRaises(KeyError):
            submit_spacetravlr.submit(DATASET, dry_run=True, partiton="typo")

    def test_flags_reach_the_command(self):
        _, _, cmd = self._capture(stages=["setup"], overwrite=True, clear_betadata=True)
        self.assertIn("--stage setup", cmd)
        self.assertIn("--overwrite", cmd)
        self.assertIn("--clear-betadata", cmd)


class TestSlurmResources(TestSubmit):
    """One job does all three stages on a GPU node. Setup is CPU work but still needs a GPU
    node -- importing the package pulls in a CUDA-only torch -- so it cannot be split onto a
    CPU partition. Memory therefore has to come from cores on the GPU partition."""

    def test_default_is_enough_cores_for_setups_peak(self):
        # ~8 GB per core on savio3_gpu; setup peaks near 50 GB on Human_Lung and OOM'd at 8.
        cfg = get_config(DATASET)["slurm"]
        self.assertGreaterEqual(cfg["cpus_per_task"], 32)

    def test_every_stage_subset_uses_the_same_gpu_profile(self):
        cfg = get_config(DATASET)["slurm"]
        for stages in (None, ["setup"], ["fit"], ["artifacts"], ["fit", "artifacts"]):
            _, sb, _ = self._capture(stages=stages)
            self.assertEqual(sb["partition"], cfg["partition"], stages)
            self.assertEqual(sb["gres"], cfg["gres"], stages)
            self.assertEqual(sb["cpus_per_task"], cfg["cpus_per_task"], stages)

    def test_no_setup_only_resource_profile_remains(self):
        self.assertNotIn("setup_slurm", get_config(DATASET))

    def test_dependency_is_passed_through_when_given(self):
        _, sb, _ = self._capture(dependency={"afterok": 123})
        self.assertEqual(sb["dependency"], {"afterok": 123})

    def test_no_dependency_key_by_default(self):
        _, sb, _ = self._capture()
        self.assertNotIn("dependency", sb)


if __name__ == "__main__":
    unittest.main()
