"""
Tier-0 pure-logic tests for `metab_processing/SpaceTravLR/metab_loader.py`: turning a
`metabolite_selection.yaml` file into the `metabolites` structure that
`SpatialCellularProgramsEstimator(metabolites=...)` consumes -- one column per
metabolite (`{column_name: [(export, import), ...]}`), with metabolites that share an
identical expanded pair-set MERGED into a single column.

No model/torch involved -- just YAML parsing + dedup/orientation/merge logic. Known-answer
assertions on a tiny inline fixture, plus a sanity pass against the real example file.
"""
import os
import sys
import unittest

import yaml

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from metab_processing.SpaceTravLR.metab_loader import (
    MERGE_SEP,
    build_metabolites,
    load_metabolites,
    load_metabolite_selection,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REAL_SELECTION_PATH = os.path.join(
    REPO_ROOT, "easy_download", "harreman_outputs", "metabolite_selection.yaml"
)

# A tiny fixture exercising: a heterotypic pair (A, B), a homotypic pair (C, C), a distinct
# metabolite (Metab2), and Metab3 whose pair-set is IDENTICAL to Metab1's (same unordered
# pairs, listed in a different order) -- so dedupe, orientation expansion, AND merge can all
# be checked precisely.
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
- name: Metab3
  gene_pairs:
  - [C, C]
  - [A, B]
"""


def _selection():
    return {
        e["name"]: [tuple(p) for p in e["gene_pairs"]]
        for e in yaml.safe_load(FIXTURE_YAML)["metabolites"]
    }


class TestLoadMetaboliteSelection(unittest.TestCase):
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
                "Metab3": [("C", "C"), ("A", "B")],
            },
        )
        for pairs in selection.values():
            for p in pairs:
                self.assertIsInstance(p, tuple)


class TestBuildMetabolites(unittest.TestCase):
    def setUp(self):
        self.selection = _selection()

    def test_default_known_answer_with_merge(self):
        """Metab1 and Metab3 have the same expanded pair-set (order-insensitive) and merge
        into one column named 'Metab1|Metab3'; Metab2 stays separate. Each column's pairs
        are the summed (export, import) list, both orientations of A-B present, homotypic
        C-C once."""
        metabolites = build_metabolites(self.selection)
        self.assertEqual(list(metabolites.keys()), [f"Metab1{MERGE_SEP}Metab3", "Metab2"])
        self.assertEqual(
            metabolites[f"Metab1{MERGE_SEP}Metab3"], [("A", "B"), ("B", "A"), ("C", "C")]
        )
        self.assertEqual(
            metabolites["Metab2"], [("A", "B"), ("B", "A"), ("D", "E"), ("E", "D")]
        )

    def test_homotypic_emitted_once_per_column(self):
        metabolites = build_metabolites(self.selection)
        self.assertEqual(metabolites[f"Metab1{MERGE_SEP}Metab3"].count(("C", "C")), 1)

    def test_pairs_not_deduped_across_columns(self):
        # (A, B)/(B, A) appears in BOTH columns -- each metabolite keeps its own copy
        # (unlike the old flat-list loader, which globally deduped).
        metabolites = build_metabolites(self.selection)
        self.assertIn(("A", "B"), metabolites[f"Metab1{MERGE_SEP}Metab3"])
        self.assertIn(("A", "B"), metabolites["Metab2"])

    def test_both_orientations_false(self):
        metabolites = build_metabolites(self.selection, both_orientations=False)
        # merge still happens (set identity is orientation-independent here)
        self.assertEqual(list(metabolites.keys()), [f"Metab1{MERGE_SEP}Metab3", "Metab2"])
        self.assertEqual(metabolites[f"Metab1{MERGE_SEP}Metab3"], [("A", "B"), ("C", "C")])
        self.assertEqual(metabolites["Metab2"], [("A", "B"), ("D", "E")])

    def test_var_names_filter_drops_pair_and_may_unmerge(self):
        # D, E absent -> Metab2's (D,E) dropped; Metab2 then has only {A-B}, which is a
        # DIFFERENT set from Metab1|Metab3's {A-B, C-C}, so it does NOT merge with them.
        metabolites = build_metabolites(self.selection, var_names=["A", "B", "C"])
        self.assertEqual(list(metabolites.keys()), [f"Metab1{MERGE_SEP}Metab3", "Metab2"])
        self.assertEqual(metabolites["Metab2"], [("A", "B"), ("B", "A")])
        self.assertEqual(
            metabolites[f"Metab1{MERGE_SEP}Metab3"], [("A", "B"), ("B", "A"), ("C", "C")]
        )

    def test_metabolite_fully_filtered_out_is_dropped(self):
        selection = {"OnlyBad": [("X", "Y")], "Good": [("A", "B")]}
        metabolites = build_metabolites(selection, var_names=["A", "B"])
        self.assertEqual(list(metabolites.keys()), ["Good"])
        self.assertEqual(metabolites["Good"], [("A", "B"), ("B", "A")])

    def test_within_metabolite_duplicate_pair_summed_once(self):
        selection = {"M": [("A", "B"), ("A", "B"), ("B", "A")]}
        metabolites = build_metabolites(selection)
        # unordered {A,B} appears three times but is one metabolite edge -> (A,B),(B,A)
        self.assertEqual(metabolites["M"], [("A", "B"), ("B", "A")])

    def test_determinism(self):
        self.assertEqual(build_metabolites(_selection()), build_metabolites(_selection()))

    def test_realistic_chemical_name_is_preserved_verbatim(self):
        # Real metabolite names carry commas, parentheses, apostrophes, hyphens, digits --
        # they must survive as the column name untouched (only the pairs are processed).
        name = "(3a,5b,7a)-23-Carboxy-7-hydroxy-24-nor'cholan-3-yl acid"
        metabolites = build_metabolites({name: [("A", "B")]})
        self.assertEqual(list(metabolites.keys()), [name])
        self.assertEqual(metabolites[name], [("A", "B"), ("B", "A")])


class TestDuplicateMetaboliteName(unittest.TestCase):
    def test_repeated_name_accumulates_pairs_no_loss(self):
        import tempfile

        yaml_text = """
metabolites:
- name: Dup
  gene_pairs:
  - [A, B]
- name: Dup
  gene_pairs:
  - [C, C]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            path = f.name
        try:
            selection = load_metabolite_selection(path)
        finally:
            os.remove(path)

        # both entries' pairs survive (no silent overwrite)
        self.assertEqual(selection, {"Dup": [("A", "B"), ("C", "C")]})
        metabolites = build_metabolites(selection)
        self.assertEqual(metabolites["Dup"], [("A", "B"), ("B", "A"), ("C", "C")])


class TestLoadMetabolitesConvenience(unittest.TestCase):
    def test_returns_metabolites_and_selection(self):
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(FIXTURE_YAML)
            path = f.name
        try:
            metabolites, selection = load_metabolites(path)
        finally:
            os.remove(path)

        self.assertEqual(len(selection), 3)  # original, ungrouped
        self.assertEqual(list(metabolites.keys()), [f"Metab1{MERGE_SEP}Metab3", "Metab2"])


@unittest.skipUnless(
    os.path.exists(REAL_SELECTION_PATH),
    f"real example file not found at {REAL_SELECTION_PATH}",
)
class TestRealMetaboliteSelectionFile(unittest.TestCase):
    def test_real_file_loads_and_builds_sanely(self):
        metabolites, selection = load_metabolites(REAL_SELECTION_PATH)

        self.assertGreater(len(selection), 0)
        self.assertGreater(len(metabolites), 0)
        # columns are named strings; merges (if any) never exceed the raw metabolite count
        self.assertLessEqual(len(metabolites), len(selection))
        for name in metabolites:
            self.assertIsInstance(name, str)

        # union of every gene mentioned anywhere in the file
        all_genes = set()
        for pairs in selection.values():
            for g1, g2 in pairs:
                all_genes.update((g1, g2))

        # every (export, import) in every column references a real gene from the file
        for pairs in metabolites.values():
            for e, i in pairs:
                self.assertIn(e, all_genes)
                self.assertIn(i, all_genes)

        # sanity: this heterotypic pair is known to be in the file (D-Glucose); both
        # orientations should appear together in whatever column carries it.
        carriers = [p for pairs in metabolites.values() for p in pairs]
        self.assertIn(("SLC2A1", "SLC2A3"), carriers)
        self.assertIn(("SLC2A3", "SLC2A1"), carriers)


if __name__ == "__main__":
    unittest.main()
