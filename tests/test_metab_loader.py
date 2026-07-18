"""
Tier-0 pure-logic tests for `metab_processing/metab_loader.py`: turning a
`metabolite_selection.yaml` file into the `metab_pairs` structure that
`SpatialCellularProgramsEstimator(metab_pairs=...)` consumes.

No model/torch involved -- just YAML parsing + dedup/orientation logic. Known-answer
assertions on a tiny inline fixture, plus a sanity pass against the real example file
in `easy_download/harreman_outputs/`.
"""
import os
import sys
import unittest

import yaml

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from metab_processing.SpaceTravLR.metab_loader import (
    build_metab_pairs,
    load_metab_pairs,
    load_metabolite_selection,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REAL_SELECTION_PATH = os.path.join(
    REPO_ROOT, "easy_download", "harreman_outputs", "metabolite_selection.yaml"
)

# A tiny fixture exercising: a heterotypic pair (A, B), a homotypic pair (C, C), and a
# pair (A, B) reappearing (same orientation) under a second metabolite -- so dedupe and
# both-orientation expansion can both be checked precisely.
FIXTURE_YAML = """
metabolites:
- name: Metab1
  gene_pairs:
  - [A, B]
  - [C, C]
- name: Metab2
  gene_pairs:
  - [A, B]
  - [D, E]
"""


class TestLoadMetaboliteSelection(unittest.TestCase):
    def test_parses_correct_structure(self):
        doc = yaml.safe_load(FIXTURE_YAML)
        # sanity on the fixture itself
        self.assertEqual([e["name"] for e in doc["metabolites"]], ["Metab1", "Metab2"])

    def test_load_from_tmp_file(self):
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(FIXTURE_YAML)
            path = f.name
        try:
            selection = load_metabolite_selection(path)
        finally:
            os.remove(path)

        self.assertEqual(
            selection,
            {
                "Metab1": [("A", "B"), ("C", "C")],
                "Metab2": [("A", "B"), ("D", "E")],
            },
        )
        # tuples, not lists
        for pairs in selection.values():
            for p in pairs:
                self.assertIsInstance(p, tuple)


class TestBuildMetabPairs(unittest.TestCase):
    def setUp(self):
        self.selection = yaml.safe_load(FIXTURE_YAML)["metabolites"]
        self.selection = {
            e["name"]: [tuple(p) for p in e["gene_pairs"]] for e in self.selection
        }

    def test_heterotypic_both_orientations_present(self):
        pairs = build_metab_pairs(self.selection)
        self.assertIn(("A", "B"), pairs)
        self.assertIn(("B", "A"), pairs)

    def test_homotypic_emitted_exactly_once_not_duplicated(self):
        pairs = build_metab_pairs(self.selection)
        self.assertEqual(pairs.count(("C", "C")), 1)

    def test_dedupe_across_metabolites(self):
        # (A, B) appears under both Metab1 and Metab2 -- should still yield (A,B) and
        # (B,A) exactly once each in the flat list, not twice.
        pairs = build_metab_pairs(self.selection)
        self.assertEqual(pairs.count(("A", "B")), 1)
        self.assertEqual(pairs.count(("B", "A")), 1)

    def test_both_orientations_default_true(self):
        pairs = build_metab_pairs(self.selection)
        # A-B heterotypic -> 2 tuples; C-C homotypic -> 1 tuple; D-E heterotypic -> 2 tuples
        self.assertEqual(len(pairs), 5)

    def test_both_orientations_false_emits_one_per_unordered_pair(self):
        pairs = build_metab_pairs(self.selection, both_orientations=False)
        self.assertIn(("A", "B"), pairs)
        self.assertNotIn(("B", "A"), pairs)
        self.assertIn(("D", "E"), pairs)
        self.assertNotIn(("E", "D"), pairs)
        # homotypic pair still exactly one
        self.assertEqual(pairs.count(("C", "C")), 1)
        self.assertEqual(len(pairs), 3)

    def test_var_names_filter_drops_pair_with_absent_gene(self):
        # D, E are not in var_names -> (D,E)/(E,D) dropped; A, B, C all present -> kept.
        var_names = ["A", "B", "C"]
        pairs = build_metab_pairs(self.selection, var_names=var_names)
        self.assertIn(("A", "B"), pairs)
        self.assertIn(("B", "A"), pairs)
        self.assertIn(("C", "C"), pairs)
        self.assertNotIn(("D", "E"), pairs)
        self.assertNotIn(("E", "D"), pairs)
        self.assertEqual(len(pairs), 3)

    def test_var_names_filter_keeps_fully_present_pair(self):
        var_names = {"A", "B", "C", "D", "E"}
        pairs = build_metab_pairs(self.selection, var_names=var_names)
        self.assertEqual(len(pairs), 5)  # nothing dropped

    def test_determinism_same_call_twice(self):
        pairs1 = build_metab_pairs(self.selection)
        pairs2 = build_metab_pairs(self.selection)
        self.assertEqual(pairs1, pairs2)

    def test_determinism_across_fresh_parses(self):
        # re-parsing the fixture from scratch should give the same flat order too.
        selection_a = {
            e["name"]: [tuple(p) for p in e["gene_pairs"]]
            for e in yaml.safe_load(FIXTURE_YAML)["metabolites"]
        }
        selection_b = {
            e["name"]: [tuple(p) for p in e["gene_pairs"]]
            for e in yaml.safe_load(FIXTURE_YAML)["metabolites"]
        }
        self.assertEqual(build_metab_pairs(selection_a), build_metab_pairs(selection_b))


class TestLoadMetabPairsConvenience(unittest.TestCase):
    def test_returns_pairs_and_selection(self):
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(FIXTURE_YAML)
            path = f.name
        try:
            pairs, selection = load_metab_pairs(path)
        finally:
            os.remove(path)

        self.assertEqual(
            selection,
            {
                "Metab1": [("A", "B"), ("C", "C")],
                "Metab2": [("A", "B"), ("D", "E")],
            },
        )
        self.assertIn(("A", "B"), pairs)
        self.assertIn(("B", "A"), pairs)
        self.assertEqual(pairs.count(("C", "C")), 1)


@unittest.skipUnless(
    os.path.exists(REAL_SELECTION_PATH),
    f"real example file not found at {REAL_SELECTION_PATH}",
)
class TestRealMetaboliteSelectionFile(unittest.TestCase):
    def test_real_file_loads_and_flattens_sanely(self):
        pairs, selection = load_metab_pairs(REAL_SELECTION_PATH)

        self.assertGreater(len(selection), 0)
        self.assertGreater(len(pairs), 0)

        # union of every gene mentioned anywhere in the file
        all_genes = set()
        for metab_pairs in selection.values():
            for g1, g2 in metab_pairs:
                all_genes.add(g1)
                all_genes.add(g2)

        for e, i in pairs:
            self.assertIn(e, all_genes)
            self.assertIn(i, all_genes)

        # sanity: this heterotypic pair is known to be in the file (D-Glucose), both
        # orientations should be present by default.
        self.assertIn(("SLC2A1", "SLC2A3"), pairs)
        self.assertIn(("SLC2A3", "SLC2A1"), pairs)


if __name__ == "__main__":
    unittest.main()
