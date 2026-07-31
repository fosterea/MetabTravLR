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
                 mock.patch.object(run_spacetravlr, "_load_adata", lambda p, s: object()):
                run_spacetravlr.run_dataset(DATASET, stages=["setup"], data_dir=tmp)

        self.assertEqual(order, ["isolate", "setup_"])


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

            def fit(self, metab_pairs=None, **kwargs):
                betadata = self.outdir / "betadata"
                before = sorted(p.name for p in betadata.glob("*_betadata.parquet")) \
                    if betadata.exists() else []
                outer.fit_calls.append(
                    {"metab_pairs": metab_pairs, "kwargs": kwargs, "betadata_before": before})
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
                    adata.uns["beta_modulators"] = {"CD4": ["ABCA1@ABCA1"]}
                    adata.obsm["beta_CD4"] = _Shape((6, 1))
                return {}
            return _fn

        patches = [
            mock.patch.object(run_spacetravlr, "SpaceShip", FakeShip),
            mock.patch.object(run_spacetravlr, "_load_adata", lambda paths, src: self.adata),
            mock.patch.object(run_spacetravlr, "load_metab_pairs",
                              lambda path, var_names=None: ([("A", "B")], {"m1": [("A", "B")]})),
            mock.patch.object(run_spacetravlr, "_processed_var_names", lambda paths: ["A", "B"]),
            mock.patch.object(run_spacetravlr.beta_analysis, "write_gene_pairs", record("write_gene_pairs")),
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

    def test_setup_lock_blocks_a_concurrent_setup(self):
        paths = make_dataset_tree(self.root)
        paths["outdir"].mkdir(parents=True, exist_ok=True)
        (paths["outdir"] / ".setup.lock").write_text("pid 1")
        with self.assertRaises(RuntimeError):
            self.run_it(stages=["setup"])
        self.assertEqual(self.setup_calls, [])

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

    def test_fit_passes_metab_pairs_and_config_kwargs(self):
        make_dataset_tree(self.root, setup=True)
        with mock.patch.dict(dataset_configs.DATASETS,
                             {DATASET: {"fit_kwargs": {"max_epochs": 3}}}, clear=False):
            self.run_it(stages=["fit"])
        self.assertEqual(len(self.fit_calls), 1)
        self.assertEqual(self.fit_calls[0]["metab_pairs"], [("A", "B")])
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
                         ["write_gene_pairs", "write_histograms", "betas_to_adata"])

    def test_artifacts_write_everything_and_the_beta_adata(self):
        paths = make_dataset_tree(self.root, setup=True, betadata_genes=["CD4"])
        self.run_it(stages=["artifacts"])
        names = [name for name, _, _ in self.artifact_calls]
        self.assertEqual(names, ["write_gene_pairs", "write_histograms", "betas_to_adata"])
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
                         ["write_gene_pairs", "write_histograms", "betas_to_adata"])
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

            # Two trained target genes, one metab pair + one TF modulator each.
            betas = pd.DataFrame(
                {
                    "beta_ABCA1@ABCA1": np.arange(6, dtype="float32"),
                    "beta_ATP7A@ATP7B": np.linspace(-1, 1, 6, dtype="float32"),
                    "beta_STAT1": np.full(6, 0.5, dtype="float32"),
                },
                index=cells,
            )
            paths["betadata"].mkdir(parents=True, exist_ok=True)
            for gene in ("CD4", "CD3E"):
                betas.to_parquet(paths["betadata"] / f"{gene}_betadata.parquet")

            run_spacetravlr.run_dataset(DATASET, stages=["artifacts"], data_dir=tmp)

            tier_dir = paths["metab_outdir"] / "Tier1"
            gene_pairs = pd.read_csv(tier_dir / "gene_pairs.csv")
            self.assertEqual(set(gene_pairs["gene"]), {"CD4", "CD3E"})
            self.assertEqual(set(gene_pairs["pair"]), {"ABCA1@ABCA1", "ATP7A@ATP7B"},
                             "gene_pairs.csv is the metab group only")
            self.assertEqual(set(gene_pairs["cell_type"]), {"T Cell", "Tumor"})

            hist = pd.read_csv(tier_dir / "histograms.csv")
            self.assertEqual(set(hist["group"]), {"metab", "tf"})
            self.assertTrue((tier_dir / "histograms.png").is_file())

            # Only Tier1 is in obs; Tier2/Tier3 must be skipped, not crash.
            self.assertFalse((paths["metab_outdir"] / "Tier2").exists())

            out = ad.read_h5ad(paths["beta_adata"])
            self.assertEqual(out.obsm["beta_CD4"].shape, (6, 3),
                             "group=None keeps every modulator group")
            self.assertEqual(list(out.uns["beta_modulators"]["CD4"]),
                             ["ABCA1@ABCA1", "ATP7A@ATP7B", "STAT1"])
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


class TestSlurmProfile(TestSubmit):
    """Setup is CPU-only but memory-hungry; training needs the GPU. Different hardware."""

    def test_setup_only_job_uses_the_big_mem_cpu_profile(self):
        _, sb, _ = self._capture(stages=["setup"])
        setup_cfg = get_config(DATASET)["setup_slurm"]
        self.assertEqual(sb["partition"], setup_cfg["partition"])
        self.assertEqual(sb["qos"], setup_cfg["qos"])
        self.assertEqual(sb["cpus_per_task"], setup_cfg["cpus_per_task"])
        self.assertEqual(sb["time"], timedelta(hours=setup_cfg["time_hours"]))
        self.assertNotIn("gres", sb, "setup must not hold a GPU")

    def test_setup_profile_inherits_unset_keys_from_the_gpu_block(self):
        _, sb, cmd = self._capture(stages=["setup"])
        cfg = get_config(DATASET)["slurm"]
        self.assertEqual(sb["account"], cfg["account"])
        self.assertTrue(cmd.startswith(cfg["python_path"]))

    def test_training_job_keeps_the_gpu_profile(self):
        _, sb, _ = self._capture(stages=["fit", "artifacts"])
        cfg = get_config(DATASET)["slurm"]
        self.assertEqual(sb["partition"], cfg["partition"])
        self.assertEqual(sb["gres"], cfg["gres"])

    def test_all_stages_in_one_job_keeps_the_gpu_profile(self):
        # The combined job cannot be both big-mem and GPU; submit_split is the answer.
        _, sb, _ = self._capture()
        self.assertEqual(sb["partition"], get_config(DATASET)["slurm"]["partition"])

    def test_setup_profile_can_still_be_overridden_per_submission(self):
        _, sb, _ = self._capture(stages=["setup"], cpus_per_task=8, time_hours=1)
        self.assertEqual(sb["cpus_per_task"], 8)
        self.assertEqual(sb["time"], timedelta(hours=1))
        self.assertEqual(sb["partition"], get_config(DATASET)["setup_slurm"]["partition"])


class TestSubmitSplit(TestSubmit):
    def _capture_split(self, **kwargs):
        calls = []
        with mock.patch("simple_slurm.Slurm") as slurm:
            slurm.return_value.sbatch.side_effect = [101, 202]
            slurm.side_effect = lambda **kw: (calls.append(kw), mock.DEFAULT)[1]
            ids = submit_spacetravlr.submit_split(DATASET, **kwargs)
        return ids, calls

    def test_chains_training_on_afterok_of_setup(self):
        (setup_id, run_id), calls = self._capture_split()
        self.assertEqual((setup_id, run_id), (101, 202))
        self.assertNotIn("dependency", calls[0], "setup job waits on nothing")
        self.assertEqual(calls[1]["dependency"], {"afterok": 101})

    def test_the_two_jobs_get_different_hardware(self):
        _, calls = self._capture_split()
        cfg = get_config(DATASET)
        self.assertEqual(calls[0]["partition"], cfg["setup_slurm"]["partition"])
        self.assertNotIn("gres", calls[0])
        self.assertEqual(calls[1]["partition"], cfg["slurm"]["partition"])
        self.assertEqual(calls[1]["gres"], cfg["slurm"]["gres"])

    def test_overwrite_goes_to_setup_and_clear_betadata_to_training(self):
        # run_dataset rejects --overwrite without the setup stage, so they must not swap.
        with mock.patch("simple_slurm.Slurm") as slurm:
            slurm.return_value.sbatch.side_effect = [1, 2]
            submit_spacetravlr.submit_split(DATASET, overwrite=True, clear_betadata=True)
        cmds = [c.args[0] for c in slurm.return_value.sbatch.call_args_list]
        self.assertIn("--overwrite", cmds[0])
        self.assertNotIn("--clear-betadata", cmds[0])
        self.assertIn("--clear-betadata", cmds[1])
        self.assertNotIn("--overwrite", cmds[1])

    def test_stages_are_split_setup_then_fit_artifacts(self):
        with mock.patch("simple_slurm.Slurm") as slurm:
            slurm.return_value.sbatch.side_effect = [1, 2]
            submit_spacetravlr.submit_split(DATASET)
        cmds = [c.args[0] for c in slurm.return_value.sbatch.call_args_list]
        self.assertIn("--stage setup", cmds[0])
        self.assertIn("--stage fit artifacts", cmds[1])

    def test_per_job_overrides_apply_to_the_right_job(self):
        _, calls = self._capture_split(setup={"time_hours": 12}, run={"time_hours": 36})
        self.assertEqual(calls[0]["time"], timedelta(hours=12))
        self.assertEqual(calls[1]["time"], timedelta(hours=36))

    def test_dry_run_submits_neither(self):
        with mock.patch("simple_slurm.Slurm") as slurm:
            ids = submit_spacetravlr.submit_split(DATASET, dry_run=True)
        self.assertEqual(ids, (None, None))
        slurm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
