"""Tier-0 tests for `metab_processing/SpaceTravLR/run_subsamples.py` -- the gene-pair
permutation-subsampling pipeline.

Only the pure functions (`sample_once`, `write_subsamples`, `pairs_to_metabolites`,
`latest_run`, `clear_markers`) and `build_run_analysis` run here, the last driven over
tiny fake betadata parquets (mirroring `tests/test_beta_analysis.py`'s fixtures). No
torch, no `SpaceShip`, no harreman -- `run_subsamples` (the SLURM job body that trains)
is exercised only via its plumbing, not here.
"""
import contextlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import anndata as ad
import numpy as np
import pandas as pd
import yaml

from metab_processing.SpaceTravLR.metab_loader import load_metabolite_selection
from metab_processing.SpaceTravLR.beta_analysis import _group
from metab_processing.SpaceTravLR.dataset_configs import dataset_paths
from metab_processing.SpaceTravLR.run_subsamples import (
    build_run_analysis,
    clear_markers,
    latest_run,
    pairs_to_metabolites,
    run_subsamples,
    sample_once,
    write_subsamples,
)

DATASET = "TestDataset"


class _AlwaysDrop:
    """Stub rng that always fails the `rng.random() < p` keep test (returns 1.0)."""

    def random(self):
        return 1.0


# ------------------------------------------------------------------------- sample_once
class SampleOnceTests(unittest.TestCase):
    def setUp(self):
        self.selection = {
            "Glucose": [("SLC2A1", "SLC2A1"), ("SLC2A1", "SLC2A9")],
            "Copper": [("ATP7A", "ATP7B")],
        }

    def test_p1_keeps_everything(self):
        rng = np.random.default_rng(0)
        out = sample_once(self.selection, 1.0, rng)
        self.assertEqual(out, self.selection)

    def test_seeded_reproducible(self):
        out1 = sample_once(self.selection, 0.5, np.random.default_rng(42))
        out2 = sample_once(self.selection, 0.5, np.random.default_rng(42))
        self.assertEqual(out1, out2)

    def test_empty_metabolite_dropped(self):
        # rng.random() sequence: keep first pair of Glucose, drop the second, drop Copper's
        # only pair -- Copper should be absent entirely, not present with an empty list.
        class Seq:
            def __init__(self, vals):
                self.vals = list(vals)

            def random(self):
                return self.vals.pop(0)

        rng = Seq([0.0, 0.9, 0.9])
        out = sample_once(self.selection, 0.5, rng)
        self.assertEqual(list(out.keys()), ["Glucose"])
        self.assertEqual(out["Glucose"], [("SLC2A1", "SLC2A1")])

    def test_stub_rng_all_drop(self):
        out = sample_once(self.selection, 0.999, _AlwaysDrop())
        self.assertEqual(out, {})


# ---------------------------------------------------------------------- write_subsamples
def _make_dataset_dir(root, dataset=DATASET, base_selection=None):
    """Skeleton `<dataset>/easy_download/harreman_outputs/sample_metabolites.yml`."""
    paths = dataset_paths(dataset, root)
    harreman_dir = paths["selection_yaml"].parent
    harreman_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "metabolites": [
            {"name": name, "gene_pairs": [list(pair) for pair in pairs]}
            for name, pairs in (base_selection or {}).items()
        ]
    }
    with open(harreman_dir / "sample_metabolites.yml", "w") as f:
        yaml.safe_dump(doc, f)
    return paths


class WriteSubsamplesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.base = {
            "Glucose": [("SLC2A1", "SLC2A1"), ("SLC2A1", "SLC2A9")],
            "Copper": [("ATP7A", "ATP7B")],
        }
        _make_dataset_dir(self.tmp, base_selection=self.base)

    def test_p_out_of_range_raises(self):
        with self.assertRaises(AssertionError):
            write_subsamples(DATASET, 1, 0.0, self.tmp, seed=0)
        with self.assertRaises(AssertionError):
            write_subsamples(DATASET, 1, 1.5, self.tmp, seed=0)

    def test_writes_n_ymls_that_round_trip(self):
        r = write_subsamples(DATASET, 3, 1.0, self.tmp, seed=0)
        self.assertEqual(r, 1)
        run_dir = self.tmp / DATASET / "easy_download" / "harreman_outputs" / "subsamples" / "run_1"
        files = sorted(run_dir.glob("sampled_metabolites_*.yml"))
        self.assertEqual(len(files), 3)
        for j, f in enumerate(files, start=1):
            self.assertEqual(f.name, f"sampled_metabolites_{j}.yml")
            selection = load_metabolite_selection(f)
            # p=1.0 -> every file should round-trip to exactly the base selection.
            self.assertEqual(selection, self.base)

    def test_run_numbering_increments(self):
        r1 = write_subsamples(DATASET, 1, 1.0, self.tmp, seed=0)
        r2 = write_subsamples(DATASET, 1, 1.0, self.tmp, seed=0)
        self.assertEqual((r1, r2), (1, 2))

    def test_never_writes_an_empty_file(self):
        # Tiny p makes most draws empty; every retry must eventually land on a non-empty
        # file (sample_once alone can legitimately return {}, write_subsamples must not
        # persist that).
        r = write_subsamples(DATASET, 5, 0.05, self.tmp, seed=1)
        run_dir = self.tmp / DATASET / "easy_download" / "harreman_outputs" / "subsamples" / f"run_{r}"
        files = sorted(run_dir.glob("sampled_metabolites_*.yml"))
        self.assertEqual(len(files), 5)
        for f in files:
            selection = load_metabolite_selection(f)
            self.assertTrue(selection, f"{f} round-tripped to an empty selection")
            for pairs in selection.values():
                self.assertTrue(pairs)


# ------------------------------------------------------------------- pairs_to_metabolites
class PairsToMetabolitesTests(unittest.TestCase):
    def test_homotypic_one_orientation(self):
        selection = {"Glucose": [("SLC2A1", "SLC2A1")]}
        out = pairs_to_metabolites(selection)
        self.assertEqual(out, {"Glucose-SLC2A1_SLC2A1": [("SLC2A1", "SLC2A1")]})

    def test_heterotypic_both_orientations(self):
        selection = {"Copper": [("ATP7A", "ATP7B")]}
        out = pairs_to_metabolites(selection)
        self.assertEqual(out, {"Copper-ATP7A_ATP7B": [("ATP7A", "ATP7B"), ("ATP7B", "ATP7A")]})

    def test_keys_are_metab_classifiable(self):
        selection = {"Glucose": [("SLC2A1", "SLC2A9")]}
        out = pairs_to_metabolites(selection)
        (key,) = out.keys()
        self.assertEqual(_group("metab@" + key), "metab")

    def test_var_names_filter_drops_missing(self):
        selection = {"Glucose": [("SLC2A1", "SLC2A9"), ("MISSING", "SLC2A1")]}
        out = pairs_to_metabolites(selection, var_names={"SLC2A1", "SLC2A9"})
        self.assertEqual(list(out.keys()), ["Glucose-SLC2A1_SLC2A9"])

    def test_var_names_filter_can_drop_whole_metabolite(self):
        selection = {"Ghost": [("MISSING1", "MISSING2")]}
        out = pairs_to_metabolites(selection, var_names={"SLC2A1"})
        self.assertEqual(out, {})

    def test_within_metabolite_dedup(self):
        # (A, B) and (B, A) are the same unordered pair -- listed twice should still
        # produce exactly one column.
        selection = {"Copper": [("ATP7A", "ATP7B"), ("ATP7B", "ATP7A")]}
        out = pairs_to_metabolites(selection)
        self.assertEqual(len(out), 1)
        self.assertEqual(out["Copper-ATP7A_ATP7B"], [("ATP7A", "ATP7B"), ("ATP7B", "ATP7A")])

    def test_multiple_pairs_each_own_column(self):
        selection = {"Glucose": [("SLC2A1", "SLC2A1"), ("SLC2A1", "SLC2A9")]}
        out = pairs_to_metabolites(selection)
        self.assertEqual(
            set(out.keys()), {"Glucose-SLC2A1_SLC2A1", "Glucose-SLC2A1_SLC2A9"}
        )


# ------------------------------------------------------------ latest_run / clear_markers
class LatestRunTests(unittest.TestCase):
    def test_missing_dir_is_zero(self):
        self.assertEqual(latest_run(Path(tempfile.mkdtemp()) / "nope"), 0)

    def test_no_runs_is_zero(self):
        d = Path(tempfile.mkdtemp())
        (d / "not_a_run").mkdir()
        self.assertEqual(latest_run(d), 0)

    def test_finds_max_run(self):
        d = Path(tempfile.mkdtemp())
        for name in ("run_1", "run_3", "run_2", "_setup"):
            (d / name).mkdir()
        self.assertEqual(latest_run(d), 3)


class ClearMarkersTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        _make_dataset_dir(self.tmp)
        self.paths = dataset_paths(DATASET, self.tmp)
        self.sub_root = self.paths["dataset_dir"] / "spacetravlr_subsamples" / "run_1"
        for j in (1, 2, 3):
            (self.sub_root / f"subsample_{j}").mkdir(parents=True)
        (self.sub_root / "subsample_1" / "DONE").write_text("done\n")
        (self.sub_root / "subsample_2" / "DONE").write_text("done\n")
        # subsample_3 has no DONE marker (still running / never finished)

    def test_removes_done_markers_explicit_run(self):
        removed = clear_markers(DATASET, run=1, data_dir=self.tmp)
        self.assertEqual(removed, 2)
        self.assertFalse((self.sub_root / "subsample_1" / "DONE").exists())
        self.assertFalse((self.sub_root / "subsample_2" / "DONE").exists())

    def test_removes_none_when_already_clear(self):
        clear_markers(DATASET, run=1, data_dir=self.tmp)
        removed_again = clear_markers(DATASET, run=1, data_dir=self.tmp)
        self.assertEqual(removed_again, 0)

    def test_run_minus_one_resolves_latest_sampling_run(self):
        # -1 resolves via the SAMPLING side's run numbering (subsamples/), not the
        # spacetravlr_subsamples tree.
        subsamples_dir = self.paths["selection_yaml"].parent / "subsamples"
        (subsamples_dir / "run_1").mkdir(parents=True)
        removed = clear_markers(DATASET, run=-1, data_dir=self.tmp)
        self.assertEqual(removed, 2)


# --------------------------------------------------------------------- build_run_analysis
class BuildRunAnalysisTests(unittest.TestCase):
    CELLS = [f"c{i}" for i in range(6)]
    BETAS = {
        "beta_metab@Glucose-SLC2A1_SLC2A1": [1.0, 2, 3, 4, 5, 6],
        "beta_metab@Copper-ATP7A_ATP7B": [0.0, 0, 0, 1, 1, 1],
        "beta_STAT1": [2.0] * 6,   # non-metab column -- must be excluded
    }

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.paths = dataset_paths(DATASET, self.tmp)
        self.paths["dataset_dir"].mkdir(parents=True, exist_ok=True)

        # a tiny raw adata.h5ad with a cell-type annotation column
        adata = ad.AnnData(np.zeros((6, 2), dtype="float32"))
        adata.obs_names = self.CELLS
        adata.obs["cell_type"] = pd.Categorical(["T Cell"] * 3 + ["other"] * 3)
        adata.write_h5ad(self.paths["adata"])

        self.sub_root = self.paths["dataset_dir"] / "spacetravlr_subsamples" / "run_1"
        for j, genes in ((1, ("CD4", "CD3E")), (2, ("CD4",))):
            betadata_dir = self.sub_root / f"subsample_{j}" / "spacetravlr_output" / "betadata"
            betadata_dir.mkdir(parents=True)
            for gene in genes:
                pd.DataFrame(self.BETAS, index=self.CELLS).to_parquet(
                    betadata_dir / f"{gene}_betadata.parquet")

    def _patch_focus_genes(self):
        """`build_run_analysis` reads focus genes from `get_config`; patch it to our
        fixture's genes so the test doesn't depend on the real dataset_configs.DATASETS.
        """
        import metab_processing.SpaceTravLR.run_subsamples as rs

        cfg = {
            "focus_genes": ["CD4", "CD3E"],
            "cell_type_src": "cell_type",
            "run_commot": False,
            "fit_kwargs": {},
            "tiers": ["cell_type"],
            "beta_group": None,
            "dataset": DATASET,
        }
        return mock.patch.object(rs, "get_config", return_value=cfg)

    def test_writes_expected_objects(self):
        with self._patch_focus_genes():
            build_run_analysis(DATASET, 1, "cell_type", data_dir=self.tmp)

        h5ad_path = self.sub_root / "subsample_betas.h5ad"
        csv_path = self.sub_root / "subsample_beta_means.csv"
        self.assertTrue(h5ad_path.exists())
        self.assertTrue(csv_path.exists())

        ra = ad.read_h5ad(h5ad_path)
        self.assertEqual(
            set(ra.obsm.keys()),
            {"beta_CD4__sample1", "beta_CD3E__sample1", "beta_CD4__sample2"},
        )
        self.assertEqual(ra.obsm["beta_CD4__sample1"].shape, (6, 2))  # metab cols only
        self.assertEqual(ra.uns["run"], 1)

        index = json.loads(ra.uns["subsample_index"])
        self.assertEqual(len(index), 3)
        by_key = {e["key"]: e for e in index}
        self.assertEqual(by_key["beta_CD4__sample1"]["gene"], "CD4")
        self.assertEqual(by_key["beta_CD4__sample1"]["sample"], 1)
        self.assertEqual(
            set(by_key["beta_CD4__sample1"]["columns"]),
            {"metab@Glucose-SLC2A1_SLC2A1", "metab@Copper-ATP7A_ATP7B"},
        )

        means = pd.read_csv(csv_path)
        self.assertEqual(list(means.columns),
                         ["sample", "gene", "cell_type", "modulator", "mean", "std", "n"])
        self.assertEqual(set(means["sample"]), {1, 2})
        # non-metab column excluded
        self.assertNotIn("STAT1", set(means["modulator"]))
        row = means[(means["sample"] == 1) & (means["gene"] == "CD4") &
                    (means["modulator"] == "metab@Glucose-SLC2A1_SLC2A1") &
                    (means["cell_type"] == "T Cell")].iloc[0]
        self.assertEqual((row["mean"], row["std"], row["n"]), (2.0, 1.0, 3))


# ------------------------------------------------------- run_subsamples (mocked, no torch)
_CELLS = [f"c{i}" for i in range(6)]
_FAKE_BETAS = {
    "beta_metab@Glucose-SLC2A1_SLC2A1": [1.0, 2, 3, 4, 5, 6],
    "beta_metab@Copper-ATP7A_ATP7B": [0.0, 0, 0, 1, 1, 1],
}


class _MockShip:
    """Stand-in for SpaceShip: records setup_/fit calls and writes the on-disk artifacts
    those stages produce, so the real orchestration (setup_is_complete, symlink, resume,
    _trained_genes, build_run_analysis) runs unchanged without torch/harreman."""

    setups: list = []
    fits: list = []

    def __init__(self, name, outdir, genes):
        self.outdir = Path(outdir)
        self.genes = genes

    def setup_(self, adata, overwrite=False, run_commot=False):
        _MockShip.setups.append(self.outdir)
        idir = self.outdir / "input_data"
        idir.mkdir(parents=True, exist_ok=True)
        a = ad.AnnData(np.zeros((3, 2), dtype="float32"))   # real h5ad so _h5ad_is_readable passes
        a.var_names = ["SLC2A1", "SLC2A9"]
        a.write_h5ad(idir / "_adata.h5ad")
        (idir / "celloracle_links.pkl").write_text("x")
        (idir / "tflinks.parquet").write_text("x")

    def fit(self, metabolites=None, **kwargs):
        _MockShip.fits.append((self.outdir, metabolites))
        bdir = self.outdir / "betadata"
        bdir.mkdir(parents=True, exist_ok=True)
        for gene in self.genes:
            pd.DataFrame(_FAKE_BETAS, index=_CELLS).to_parquet(bdir / f"{gene}_betadata.parquet")


class RunSubsamplesTests(unittest.TestCase):
    def setUp(self):
        _MockShip.setups, _MockShip.fits = [], []
        self.tmp = Path(tempfile.mkdtemp())
        self.base = {
            "Glucose": [("SLC2A1", "SLC2A1"), ("SLC2A1", "SLC2A9")],
            "Copper": [("ATP7A", "ATP7B")],
        }
        self.paths = _make_dataset_dir(self.tmp, base_selection=self.base)
        self.paths["dataset_dir"].mkdir(parents=True, exist_ok=True)
        # raw adata.h5ad for build_run_analysis (obs + cell-type col)
        adata = ad.AnnData(np.zeros((6, 2), dtype="float32"))
        adata.obs_names = _CELLS
        adata.obs["cell_type"] = pd.Categorical(["T Cell"] * 3 + ["other"] * 3)
        adata.write_h5ad(self.paths["adata"])
        # two sampled subsamples in run_1 (p=1 -> both equal the base selection)
        self.run = write_subsamples(DATASET, 2, 1.0, self.tmp, seed=0)
        self.sub_root = self.paths["dataset_dir"] / "spacetravlr_subsamples" / f"run_{self.run}"

    @contextlib.contextmanager
    def _patched(self):
        import metab_processing.SpaceTravLR.run_subsamples as rs
        cfg = {"focus_genes": ["CD4", "CD3E"], "cell_type_src": "cell_type",
               "run_commot": False, "fit_kwargs": {}, "tiers": ["cell_type"],
               "beta_group": None, "dataset": DATASET}
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(rs, "SpaceShip", _MockShip))
            stack.enter_context(mock.patch.object(rs, "get_config", return_value=cfg))
            stack.enter_context(mock.patch.object(rs, "_isolate_cache_dir", lambda: None))
            stack.enter_context(mock.patch.object(rs, "_load_adata",
                                                  lambda paths, col: ad.AnnData(np.zeros((3, 2), dtype="float32"))))
            stack.enter_context(mock.patch.object(rs, "_drop_tiny_clusters", lambda a, col: a))
            stack.enter_context(mock.patch.object(
                rs, "_processed_var_names",
                lambda paths: ["SLC2A1", "SLC2A9", "ATP7A", "ATP7B"]))
            yield

    def test_full_run_fits_all_and_builds_analysis(self):
        with self._patched():
            run_subsamples(DATASET, run=-1, data_dir=self.tmp)
        # setup once (shared), fit once per subsample
        self.assertEqual(len(_MockShip.setups), 1)
        self.assertEqual(len(_MockShip.fits), 2)
        # the fit got the per-pair metabolite columns from pairs_to_metabolites
        _outdir, metabolites = _MockShip.fits[0]
        self.assertIn("Glucose-SLC2A1_SLC2A1", metabolites)
        self.assertIn("Copper-ATP7A_ATP7B", metabolites)
        # DONE markers + shared input_data symlinked into each subsample
        for j in (1, 2):
            self.assertTrue((self.sub_root / f"subsample_{j}" / "DONE").exists())
            link = self.sub_root / f"subsample_{j}" / "spacetravlr_output" / "input_data"
            self.assertTrue(link.is_symlink())
        self.assertTrue((self.sub_root / "subsample_betas.h5ad").exists())
        self.assertTrue((self.sub_root / "subsample_beta_means.csv").exists())

    def test_resume_skips_done_subsample(self):
        (self.sub_root / "subsample_1").mkdir(parents=True)
        (self.sub_root / "subsample_1" / "DONE").write_text("done\n")
        with self._patched():
            run_subsamples(DATASET, run=1, data_dir=self.tmp)
        # only subsample_2 is fit; subsample_1 is skipped by its marker
        self.assertEqual(len(_MockShip.fits), 1)
        self.assertTrue(str(_MockShip.fits[0][0]).endswith("subsample_2/spacetravlr_output"))

    def test_second_run_is_a_noop_and_reuses_setup(self):
        with self._patched():
            run_subsamples(DATASET, run=1, data_dir=self.tmp)
            _MockShip.setups, _MockShip.fits = [], []
            run_subsamples(DATASET, run=1, data_dir=self.tmp)   # all DONE, setup complete
        self.assertEqual(_MockShip.setups, [])   # shared setup already complete -> skipped
        self.assertEqual(_MockShip.fits, [])     # every subsample already DONE

    def test_overwrite_rebuilds_setup_but_keeps_done(self):
        with self._patched():
            run_subsamples(DATASET, run=1, data_dir=self.tmp)
            _MockShip.setups, _MockShip.fits = [], []
            run_subsamples(DATASET, run=1, overwrite=True, data_dir=self.tmp)
        self.assertEqual(len(_MockShip.setups), 1)   # setup rebuilt
        self.assertEqual(_MockShip.fits, [])         # DONE markers kept -> no re-fit


if __name__ == "__main__":
    unittest.main()
