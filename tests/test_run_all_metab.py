"""Tier-0 tests for `metab_processing/SpaceTravLR/run_all_metab.py` -- the ALL-METAB test
harness that sources metabolites from `metabolite_no_selection.yaml` (every harreman
transporter pair, not just the FDR-significant ones) and writes to parallel `all_metab_*`
output paths, so a normal `run_spacetravlr.py` run is never touched.

No model/torch involved -- just path-dict rewriting and YAML-dict parsing.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Reuse test_spacetravlr_runner's dataset-tree/mocking helpers instead of reinventing them.
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from metab_processing.SpaceTravLR import run_all_metab, run_spacetravlr
from metab_processing.SpaceTravLR.dataset_configs import dataset_paths
from test_spacetravlr_runner import write_tiny_h5ad

DATASET = "Primary_Dermal_Melanoma"

# The 6 keys `all_metab_paths` is specified to rewrite; everything else must pass through.
REWRITTEN_KEYS = {
    "selection_yaml", "outdir", "input_data", "betadata", "metab_outdir", "beta_adata",
}


class TestAllMetabPaths(unittest.TestCase):
    def test_rewrites_exactly_the_expected_keys(self):
        normal = dataset_paths(DATASET, "/data")
        all_metab = run_all_metab.all_metab_paths(DATASET, "/data")

        self.assertEqual(set(all_metab), set(normal), "same key set as the normal paths dict")
        changed = {k for k in normal if all_metab[k] != normal[k]}
        self.assertEqual(changed, REWRITTEN_KEYS)

    def test_adata_dataset_dir_and_log_dir_are_untouched(self):
        normal = dataset_paths(DATASET, "/data")
        all_metab = run_all_metab.all_metab_paths(DATASET, "/data")
        for key in ("adata", "dataset_dir", "log_dir"):
            self.assertEqual(all_metab[key], normal[key], key)

    def test_selection_yaml_points_at_no_selection_file(self):
        paths = run_all_metab.all_metab_paths(DATASET, "/data")
        self.assertEqual(
            paths["selection_yaml"],
            Path("/data") / DATASET / "easy_download" / "harreman_outputs"
            / "metabolite_no_selection.yaml",
        )

    def test_outdir_and_children_are_parallel_all_metab_paths(self):
        paths = run_all_metab.all_metab_paths(DATASET, "/data")
        self.assertEqual(paths["outdir"],
                         Path("/data") / DATASET / "all_metab_spacetravlr_output")
        # input_data/betadata must be derived from the NEW outdir, not the old one.
        self.assertEqual(paths["input_data"], paths["outdir"] / "input_data")
        self.assertEqual(paths["betadata"], paths["outdir"] / "betadata")
        self.assertFalse(str(paths["input_data"]).startswith(
            str(Path("/data") / DATASET / "spacetravlr_output")))

    def test_metab_outdir_and_beta_adata_are_parallel_all_metab_paths(self):
        paths = run_all_metab.all_metab_paths(DATASET, "/data")
        self.assertEqual(
            paths["metab_outdir"],
            Path("/data") / DATASET / "easy_download" / "all_metab_metabtravlr_outputs")
        self.assertEqual(
            paths["beta_adata"], Path("/data") / DATASET / "all_metab_spacetravlr_adata.h5ad")

    def test_does_not_collide_with_the_normal_run_s_paths(self):
        normal = dataset_paths(DATASET, "/data")
        all_metab = run_all_metab.all_metab_paths(DATASET, "/data")
        for key in REWRITTEN_KEYS:
            self.assertNotEqual(all_metab[key], normal[key], key)


# A tiny fixture in the no-selection DICT format:
# {"metabolites": {name: {"gene_pair": [[g1,g2],...], "gene_type": [...]}}}
# -- Metab1 has a heterotypic pair (A,B) and a homotypic pair (C,C); Metab2 has a pair
# (D,E) where E is absent from var_names, to exercise var-name filtering.
NO_SELECTION_YAML = """
metabolites:
  Metab1:
    gene_pair:
    - [A, B]
    - [C, C]
    gene_type:
    - [IMP-EXP, IMP-EXP]
    - [IMP-EXP, IMP-EXP]
  Metab2:
    gene_pair:
    - [D, E]
    gene_type:
    - [IMP-EXP, IMP-EXP]
"""


def _write(text):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    f.write(text)
    f.close()
    return f.name


class TestLoadNoSelectionMetabolites(unittest.TestCase):
    def test_parses_dict_format_known_answer(self):
        path = _write(NO_SELECTION_YAML)
        try:
            metabolites, selection = run_all_metab.load_no_selection_metabolites(path)
        finally:
            os.remove(path)

        # `selection` is the raw {name: [(g1,g2), ...]} grouping, verbatim from the dict.
        self.assertEqual(selection, {
            "Metab1": [("A", "B"), ("C", "C")],
            "Metab2": [("D", "E")],
        })
        # `metabolites` is the expanded {column: [(export, import), ...]} the estimator
        # wants: heterotypic pair gets both orientations, homotypic pair appears once.
        self.assertEqual(metabolites["Metab1"], [("A", "B"), ("B", "A"), ("C", "C")])
        self.assertEqual(metabolites["Metab2"], [("D", "E"), ("E", "D")])

    def test_var_names_filter_drops_a_pair_and_can_drop_a_metabolite_entirely(self):
        path = _write(NO_SELECTION_YAML)
        try:
            metabolites, selection = run_all_metab.load_no_selection_metabolites(
                path, var_names=["A", "B", "C"])
        finally:
            os.remove(path)

        # Metab2's only pair (D,E) references genes absent from var_names -> dropped whole.
        self.assertNotIn("Metab2", metabolites)
        self.assertEqual(metabolites["Metab1"], [("A", "B"), ("B", "A"), ("C", "C")])

    def test_both_orientations_false(self):
        path = _write(NO_SELECTION_YAML)
        try:
            metabolites, _ = run_all_metab.load_no_selection_metabolites(
                path, both_orientations=False)
        finally:
            os.remove(path)
        self.assertEqual(metabolites["Metab1"], [("A", "B"), ("C", "C")])

    def test_entry_missing_gene_pair_key_is_treated_as_empty(self):
        # A malformed/partial entry must not crash the loader -- it just contributes no pairs
        # and (since it ends up empty) is dropped by build_metabolites.
        path = _write("metabolites:\n  Weird:\n    gene_type: [[IMP-EXP, IMP-EXP]]\n")
        try:
            metabolites, selection = run_all_metab.load_no_selection_metabolites(path)
        finally:
            os.remove(path)
        self.assertEqual(selection, {"Weird": []})
        self.assertNotIn("Weird", metabolites)

    def test_entry_value_not_a_dict_is_treated_as_empty(self):
        path = _write("metabolites:\n  Weird: null\n")
        try:
            metabolites, selection = run_all_metab.load_no_selection_metabolites(path)
        finally:
            os.remove(path)
        self.assertEqual(selection, {"Weird": []})
        self.assertNotIn("Weird", metabolites)

    def test_empty_metabolites_dict(self):
        path = _write("metabolites: {}\n")
        try:
            metabolites, selection = run_all_metab.load_no_selection_metabolites(path)
        finally:
            os.remove(path)
        self.assertEqual(selection, {})
        self.assertEqual(metabolites, {})


class TestDictFormatVsListFormatGuard(unittest.TestCase):
    """Documents WHY this module exists rather than reusing `metab_loader.load_metabolites`
    directly: `metabolite_no_selection.yaml` is a DICT keyed by metabolite name
    (`{"metabolites": {name: {"gene_pair": [...]}}}`), but the normal loader expects a LIST
    (`{"metabolites": [{"name": ..., "gene_pairs": [...]}]}`) and silently reads zero
    metabolites out of the dict format (`.get("metabolites", []) or []` iterates a dict's
    keys, i.e. plain strings, and `entry["name"]` then raises)."""

    def test_normal_load_metabolite_selection_cannot_parse_the_dict_format(self):
        from metab_processing.SpaceTravLR.metab_loader import load_metabolite_selection

        path = _write(NO_SELECTION_YAML)
        try:
            with self.assertRaises(TypeError):
                load_metabolite_selection(path)
        finally:
            os.remove(path)


class TestRunAllMetabEndToEnd(unittest.TestCase):
    """Proves the monkeypatch inside `run_all_metab.main()` actually reaches
    `run_spacetravlr.run_dataset` -- not just that `all_metab_paths` and
    `load_no_selection_metabolites` are individually correct in isolation (the tests
    above), but that calling `run_all_metab.main()` makes `run_dataset` read
    `metabolite_no_selection.yaml` (DICT format) from the `all_metab_spacetravlr_output/`
    tree instead of the normal run's `spacetravlr_output/` + `metabolite_selection.yaml`.

    Guards against: if `run_spacetravlr.py` is ever refactored from
    `from ...dataset_configs import dataset_paths` (a bare module-global, reassignable at
    runtime) to `dataset_configs.dataset_paths(...)` (an attribute lookup that ignores a
    module-level reassignment), `run_spacetravlr.dataset_paths = all_metab_paths` becomes a
    silent no-op: `run_dataset` keeps reading the *normal* paths, and the isolated unit
    tests above -- which never call `run_all_metab.main()` -- would stay green regardless.

    Only the `fit` stage is exercised (no torch/scanpy training): `SpaceShip` is a fake
    that just records what it was called with, and `_processed_var_names` is stubbed so we
    don't need a real processed adata. This keeps the test Tier-0.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name

        # `run_all_metab.main()` reassigns these two module globals on `run_spacetravlr`
        # and (by design -- see run_all_metab.py's docstring) never restores them, since
        # each real invocation is a fresh SLURM-job subprocess. A test process is not, so
        # restore them ourselves afterward to avoid leaking the redirect into any other
        # test that imports/uses `run_spacetravlr` in this same session.
        self.addCleanup(setattr, run_spacetravlr, "dataset_paths", run_spacetravlr.dataset_paths)
        self.addCleanup(setattr, run_spacetravlr, "load_metabolites", run_spacetravlr.load_metabolites)

        self.fit_calls = []
        outer = self

        class FakeShip:
            def __init__(self, name=None, outdir=None, genes=None):
                self.outdir = Path(outdir)
                self.genes = genes

            def fit(self, metabolites=None, **kwargs):
                outer.fit_calls.append(
                    {"outdir": self.outdir, "metabolites": metabolites, "kwargs": kwargs})
                betadata = self.outdir / "betadata"
                betadata.mkdir(parents=True, exist_ok=True)
                for gene in self.genes:
                    (betadata / f"{gene}_betadata.parquet").write_text("x")

        self.var_names = ["A", "B", "C", "D", "E"]
        patches = [
            mock.patch.object(run_spacetravlr, "SpaceShip", FakeShip),
            mock.patch.object(run_spacetravlr, "_processed_var_names", lambda paths: self.var_names),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def _build_all_metab_tree(self):
        """Only the ALL-METAB (`all_metab_*`) paths are populated: the shared `adata.h5ad`
        placeholder, a DICT-format `metabolite_no_selection.yaml` (NOT the normal
        list-format `metabolite_selection.yaml`), and a completed "setup" under
        `all_metab_spacetravlr_output/` (never under the normal `spacetravlr_output/`).
        The normal run's tree is deliberately left absent, so a run that (due to a
        regression) fell back to the normal paths would fail loudly with
        `FileNotFoundError` instead of silently succeeding against the wrong tree.
        """
        paths = run_all_metab.all_metab_paths(DATASET, self.root)
        paths["dataset_dir"].mkdir(parents=True, exist_ok=True)
        paths["adata"].write_text("not really an h5ad")   # never opened for --stage fit

        paths["selection_yaml"].parent.mkdir(parents=True, exist_ok=True)
        paths["selection_yaml"].write_text(NO_SELECTION_YAML)
        self.assertEqual(paths["selection_yaml"].name, "metabolite_no_selection.yaml")

        paths["input_data"].mkdir(parents=True, exist_ok=True)
        write_tiny_h5ad(paths["input_data"] / "_adata.h5ad", var_names=self.var_names)
        for name in ("celloracle_links.pkl", "tflinks.parquet"):
            (paths["input_data"] / name).write_text("x")
        return paths

    def test_main_redirects_run_dataset_to_all_metab_paths_and_dict_loader(self):
        all_metab = self._build_all_metab_tree()

        run_all_metab.main(["--dataset", DATASET, "--stage", "fit", "--data-dir", self.root])

        # (a) run_dataset read from / wrote under the all_metab_* tree, not the normal one.
        self.assertEqual(len(self.fit_calls), 1)
        self.assertEqual(self.fit_calls[0]["outdir"], all_metab["outdir"])
        self.assertTrue(str(all_metab["outdir"]).endswith("all_metab_spacetravlr_output"))
        written = sorted(p.name for p in all_metab["betadata"].glob("*_betadata.parquet"))
        self.assertTrue(written, "betadata should have been written under all_metab_spacetravlr_output/")

        normal_paths = dataset_paths(DATASET, self.root)
        self.assertFalse(normal_paths["betadata"].exists(),
                         "the normal spacetravlr_output/ tree must be untouched")
        self.assertFalse(normal_paths["selection_yaml"].exists(),
                         "the normal metabolite_selection.yaml was never created by this test, "
                         "so a run that read it (instead of the no-selection file) would have "
                         "raised FileNotFoundError above rather than reaching this assertion")

        # (b) the DICT-format no-selection loader ran, not the normal LIST-format loader
        # (which raises TypeError on this exact fixture -- see
        # TestDictFormatVsListFormatGuard above). Known-answer: both orientations expanded,
        # the homotypic pair collapsed to one.
        self.assertEqual(self.fit_calls[0]["metabolites"], {
            "Metab1": [("A", "B"), ("B", "A"), ("C", "C")],
            "Metab2": [("D", "E"), ("E", "D")],
        })


if __name__ == "__main__":
    unittest.main()
