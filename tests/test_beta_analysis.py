"""Tier-0 pure-logic tests for `metab_processing/beta_analysis.py`: the read-out side of
MetabTravLR (reading trained metabolite betas back out of betadata parquet files and
aggregating them into metabolite x gene[, cell_type] and gene-set scores).

No model/torch involved -- tiny hand-built betadata parquets with KNOWN beta values and
known-answer assertions, plus a tiny fake `gene_pair_summary.csv` for the C_np-weights
helper.
"""
import math
import os
import sys
import tempfile
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np
import pandas as pd

from metab_processing.beta_analysis import (
    aggregate_to_metabolite,
    gene_pair_cnp_weights,
    gene_set_score,
    read_metab_beta_summary,
)


def _sample_std(values):
    """Reference ddof=1 sample std, computed independently of pandas' .std() call path
    used inside beta_analysis, for known-answer assertions."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = arr.mean()
    return math.sqrt(((arr - mean) ** 2).sum() / (n - 1))


class TestReadMetabBetaSummary(unittest.TestCase):
    """Builds two tiny betadata parquets (genes GA, GB) with hand-chosen beta_<e>@<i>
    columns plus non-metabolite columns (beta0, beta_TF, beta_L$R) that must be ignored."""

    CELLS = ["c0", "c1", "c2", "c3"]

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.betadata_dir = self.tmpdir.name

        ga = pd.DataFrame(
            {
                "beta0": [0.1, 0.2, 0.3, 0.4],
                "beta_TF1": [1.0, 1.0, 1.0, 1.0],
                "beta_L$R": [5.0, 5.0, 5.0, 5.0],       # L-R, no '@' -> must be ignored
                "beta_A@B": [1.0, 2.0, 3.0, 4.0],
                "beta_B@A": [0.0, 0.0, 3.0, 3.0],
                "beta_C@C": [2.0, 2.0, 2.0, 2.0],
                "beta_D@D": [10.0, 10.0, 10.0, 10.0],
            },
            index=self.CELLS,
        )
        ga.to_parquet(os.path.join(self.betadata_dir, "GA_betadata.parquet"))

        gb = pd.DataFrame(
            {
                "beta0": [0.0, 0.0, 0.0, 0.0],
                "beta_TF2": [1.0, 1.0, 1.0, 1.0],
                "beta_A@B": [10.0, 20.0, 10.0, 20.0],   # GB is missing B@A/C@C/D@D
            },
            index=self.CELLS,
        )
        gb.to_parquet(os.path.join(self.betadata_dir, "GB_betadata.parquet"))

        self.obs = pd.DataFrame({"ct": ["ct1", "ct1", "ct2", "ct2"]}, index=self.CELLS)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _row(self, df, gene, pair, cell_type=None):
        mask = (df["gene"] == gene) & (df["pair"] == pair)
        if cell_type is not None:
            mask &= df["cell_type"] == cell_type
        sub = df[mask]
        self.assertEqual(len(sub), 1, f"expected exactly one row for {gene}/{pair}/{cell_type}")
        return sub.iloc[0]

    def test_only_metabolite_columns_picked_up(self):
        out = read_metab_beta_summary(self.betadata_dir, genes=["GA", "GB"])
        # GA contributes 4 metab pairs, GB contributes 1 -> 5 rows total; beta0/TF/L$R absent
        self.assertEqual(len(out), 5)
        self.assertEqual(set(out.loc[out["gene"] == "GA", "pair"]), {"A@B", "B@A", "C@C", "D@D"})
        self.assertEqual(set(out.loc[out["gene"] == "GB", "pair"]), {"A@B"})

    def test_export_import_parsing(self):
        out = read_metab_beta_summary(self.betadata_dir, genes=["GA"])
        row = self._row(out, "GA", "A@B")
        self.assertEqual(row["export"], "A")
        self.assertEqual(row["import"], "B")

    def test_known_mean_std_n_frac_nonzero(self):
        out = read_metab_beta_summary(self.betadata_dir, genes=["GA", "GB"])

        row = self._row(out, "GA", "A@B")
        self.assertAlmostEqual(row["mean"], 2.5)
        self.assertAlmostEqual(row["std"], _sample_std([1, 2, 3, 4]))
        self.assertEqual(row["n"], 4)
        self.assertAlmostEqual(row["frac_nonzero"], 1.0)

        row = self._row(out, "GA", "B@A")
        self.assertAlmostEqual(row["mean"], 1.5)
        self.assertAlmostEqual(row["std"], _sample_std([0, 0, 3, 3]))
        self.assertAlmostEqual(row["frac_nonzero"], 0.5)  # 2 of 4 cells nonzero

        row = self._row(out, "GA", "C@C")
        self.assertAlmostEqual(row["mean"], 2.0)
        self.assertAlmostEqual(row["std"], 0.0)
        self.assertAlmostEqual(row["frac_nonzero"], 1.0)

        row = self._row(out, "GB", "A@B")
        self.assertAlmostEqual(row["mean"], 15.0)
        self.assertAlmostEqual(row["std"], _sample_std([10, 20, 10, 20]))
        self.assertAlmostEqual(row["frac_nonzero"], 1.0)

    def test_genes_filter_skips_missing_gene(self):
        out = read_metab_beta_summary(self.betadata_dir, genes=["GA", "NOPE"])
        self.assertEqual(set(out["gene"]), {"GA"})

    def test_cell_type_grouping(self):
        out = read_metab_beta_summary(
            self.betadata_dir, genes=["GA"], obs=self.obs, cell_type_col="ct"
        )
        self.assertIn("cell_type", out.columns)

        row = self._row(out, "GA", "A@B", "ct1")
        self.assertAlmostEqual(row["mean"], 1.5)  # cells c0,c1 -> [1,2]
        self.assertEqual(row["n"], 2)

        row = self._row(out, "GA", "A@B", "ct2")
        self.assertAlmostEqual(row["mean"], 3.5)  # cells c2,c3 -> [3,4]

        row = self._row(out, "GA", "B@A", "ct1")
        self.assertAlmostEqual(row["mean"], 0.0)  # [0,0]
        self.assertAlmostEqual(row["frac_nonzero"], 0.0)

        row = self._row(out, "GA", "B@A", "ct2")
        self.assertAlmostEqual(row["mean"], 3.0)  # [3,3]
        self.assertAlmostEqual(row["frac_nonzero"], 1.0)

    def test_no_cell_type_grouping_without_both_args(self):
        # obs given but cell_type_col missing -> pooled (no cell_type column)
        out = read_metab_beta_summary(self.betadata_dir, genes=["GA"], obs=self.obs)
        self.assertNotIn("cell_type", out.columns)


class TestAggregateToMetabolite(unittest.TestCase):
    def setUp(self):
        # Pre-built pair_summary mimicking read_metab_beta_summary's pooled output
        # (as if from GA: A@B=2.5, B@A=1.5, C@C=2.0, D@D=10.0; GB: A@B=15.0 only).
        self.pair_summary = pd.DataFrame(
            [
                {"gene": "GA", "export": "A", "import": "B", "pair": "A@B", "mean": 2.5},
                {"gene": "GA", "export": "B", "import": "A", "pair": "B@A", "mean": 1.5},
                {"gene": "GA", "export": "C", "import": "C", "pair": "C@C", "mean": 2.0},
                {"gene": "GA", "export": "D", "import": "D", "pair": "D@D", "mean": 10.0},
                {"gene": "GB", "export": "A", "import": "B", "pair": "A@B", "mean": 15.0},
            ]
        )
        self.selection = {
            "Metab_AB": [("A", "B")],
            "Metab_C": [("C", "C")],
            "Metab_CD": [("C", "C"), ("D", "D")],
        }

    def _score(self, out, metab, gene):
        sub = out[(out["metabolite"] == metab) & (out["gene"] == gene)]
        self.assertEqual(len(sub), 1, f"expected exactly one row for {metab}/{gene}")
        return sub.iloc[0]

    def test_unweighted_orientation_agnostic_and_missing_pair(self):
        out = aggregate_to_metabolite(self.pair_summary, self.selection)

        # Metab_AB/GA matches BOTH A@B (2.5) and B@A (1.5) -> mean 2.0, n_pairs=2
        row = self._score(out, "Metab_AB", "GA")
        self.assertAlmostEqual(row["score"], 2.0)
        self.assertEqual(row["n_pairs"], 2)

        # Metab_AB/GB: only A@B present for GB (B@A absent from betadata) -> not counted,
        # so GB's Metab_AB score is just the one available row.
        row = self._score(out, "Metab_AB", "GB")
        self.assertAlmostEqual(row["score"], 15.0)
        self.assertEqual(row["n_pairs"], 1)

        # Metab_C/GA: homotypic C@C -> mean 2.0, n_pairs=1
        row = self._score(out, "Metab_C", "GA")
        self.assertAlmostEqual(row["score"], 2.0)
        self.assertEqual(row["n_pairs"], 1)

        # Metab_C/GB: GB has no C@C column at all -> no row emitted
        sub = out[(out["metabolite"] == "Metab_C") & (out["gene"] == "GB")]
        self.assertEqual(len(sub), 0)

        # Metab_CD/GA: C@C (2.0) + D@D (10.0) -> mean 6.0, n_pairs=2
        row = self._score(out, "Metab_CD", "GA")
        self.assertAlmostEqual(row["score"], 6.0)
        self.assertEqual(row["n_pairs"], 2)

    def test_weighted_mean_known_answer(self):
        # (A,B) pair intentionally absent from weights -> Metab_AB should be skipped
        # entirely (not just zero-weighted).
        weights = {frozenset(("C", "C")): 1.0, frozenset(("D", "D")): 4.0}
        out = aggregate_to_metabolite(self.pair_summary, self.selection, weights=weights)

        sub = out[out["metabolite"] == "Metab_AB"]
        self.assertEqual(len(sub), 0, "pair absent from weights must be fully skipped")

        row = self._score(out, "Metab_C", "GA")
        self.assertAlmostEqual(row["score"], 2.0)  # only one weighted pair -> weight cancels
        self.assertEqual(row["n_pairs"], 1)

        # weighted mean = (2.0*1 + 10.0*4) / (1+4) = 8.4  (vs unweighted 6.0)
        row = self._score(out, "Metab_CD", "GA")
        self.assertAlmostEqual(row["score"], 8.4)
        self.assertEqual(row["n_pairs"], 2)

    def test_empty_pair_summary(self):
        out = aggregate_to_metabolite(self.pair_summary.iloc[0:0], self.selection)
        self.assertEqual(len(out), 0)
        self.assertIn("metabolite", out.columns)
        self.assertIn("score", out.columns)

    def test_cell_type_column_propagated(self):
        ps = self.pair_summary.copy()
        ps["cell_type"] = "ctA"
        out = aggregate_to_metabolite(ps, {"Metab_C": [("C", "C")]})
        self.assertIn("cell_type", out.columns)
        row = out[(out["metabolite"] == "Metab_C") & (out["gene"] == "GA")].iloc[0]
        self.assertEqual(row["cell_type"], "ctA")


class TestGeneSetScore(unittest.TestCase):
    def setUp(self):
        self.metab_summary = pd.DataFrame(
            [
                {"metabolite": "M1", "gene": "G1", "score": 1.0},
                {"metabolite": "M1", "gene": "G2", "score": 3.0},
                {"metabolite": "M4", "gene": "G1", "score": 20.0},
                {"metabolite": "M4", "gene": "G2", "score": 11.0},
                {"metabolite": "M0", "gene": "G1", "score": 0.0},
                {"metabolite": "M0", "gene": "G2", "score": 0.0},
                {"metabolite": "M2", "gene": "G1", "score": 5.0},
                {"metabolite": "M2", "gene": "G2", "score": np.nan},
                {"metabolite": "M3", "gene": "G1", "score": 2.0},  # no G2 row at all
            ]
        )

    def test_signed_score_known_answer_and_sort(self):
        out = gene_set_score(self.metab_summary, {"positive": ["G1"], "negative": ["G2"]})

        self.assertIn("signed", out.columns)
        expected = out.set_index("metabolite")
        self.assertAlmostEqual(expected.loc["M1", "signed"], 1.0 - 3.0)
        self.assertAlmostEqual(expected.loc["M4", "signed"], 20.0 - 11.0)
        self.assertAlmostEqual(expected.loc["M0", "signed"], 0.0)
        self.assertTrue(math.isnan(expected.loc["M2", "signed"]))  # negative missing -> NaN
        self.assertTrue(math.isnan(expected.loc["M3", "signed"]))  # negative absent entirely

        # descending by signed, NaNs last, stable order among the NaN pair (M2 before M3)
        self.assertEqual(list(out["metabolite"]), ["M4", "M0", "M1", "M2", "M3"])

    def test_single_label_sort_and_no_signed_column(self):
        out = gene_set_score(self.metab_summary, {"activity": ["G1"]})
        self.assertNotIn("signed", out.columns)
        self.assertIn("activity", out.columns)
        # descending by activity: M4(20) > M2(5) > M3(2) > M1(1) > M0(0)
        self.assertEqual(list(out["metabolite"]), ["M4", "M2", "M3", "M1", "M0"])

    def test_missing_gene_in_label_does_not_crash(self):
        out = gene_set_score(self.metab_summary, {"positive": ["G1"], "negative": ["NOT_A_GENE"]})
        # 'NOT_A_GENE' present in nobody -> negative is NaN everywhere, no crash
        self.assertTrue(out["negative"].isna().all())
        self.assertTrue(out["signed"].isna().all())


class TestGenePairCnpWeights(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.csv_rows = pd.DataFrame(
            [
                {"tier": "Tier1", "gene1": "A", "gene2": "B", "C_np": 10.0},
                {"tier": "Tier1", "gene1": "B", "gene2": "A", "C_np": 15.0},  # same unordered pair
                {"tier": "Tier2", "gene1": "A", "gene2": "B", "C_np": 99.0},  # different tier
                {"tier": "Tier1", "gene1": "C", "gene2": "C", "C_np": 5.0},   # homotypic
            ]
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_max_agg_direct_csv_path(self):
        csv_path = os.path.join(self.tmpdir.name, "gene_pair_summary.csv")
        self.csv_rows.to_csv(csv_path, index=False)

        weights = gene_pair_cnp_weights(csv_path, tier="Tier1", agg="max")
        self.assertEqual(
            weights,
            {frozenset(("A", "B")): 15.0, frozenset(("C",)): 5.0},
        )

    def test_mean_agg(self):
        csv_path = os.path.join(self.tmpdir.name, "gene_pair_summary.csv")
        self.csv_rows.to_csv(csv_path, index=False)

        weights = gene_pair_cnp_weights(csv_path, tier="Tier1", agg="mean")
        self.assertAlmostEqual(weights[frozenset(("A", "B"))], 12.5)
        self.assertAlmostEqual(weights[frozenset(("C",))], 5.0)

    def test_folder_with_summary_subdir(self):
        # easy_download_path points at a folder containing summary/gene_pair_summary.csv
        summary_dir = os.path.join(self.tmpdir.name, "summary")
        os.makedirs(summary_dir, exist_ok=True)
        self.csv_rows.to_csv(os.path.join(summary_dir, "gene_pair_summary.csv"), index=False)

        weights = gene_pair_cnp_weights(self.tmpdir.name, tier="Tier1", agg="max")
        self.assertEqual(weights[frozenset(("A", "B"))], 15.0)

    def test_no_tier_filter_pools_all_rows(self):
        csv_path = os.path.join(self.tmpdir.name, "gene_pair_summary.csv")
        self.csv_rows.to_csv(csv_path, index=False)

        weights = gene_pair_cnp_weights(csv_path, tier=None, agg="max")
        # (A,B) now also sees the Tier2 row (99.0)
        self.assertEqual(weights[frozenset(("A", "B"))], 99.0)

    def test_invalid_agg_raises(self):
        csv_path = os.path.join(self.tmpdir.name, "gene_pair_summary.csv")
        self.csv_rows.to_csv(csv_path, index=False)
        with self.assertRaises(ValueError):
            gene_pair_cnp_weights(csv_path, agg="bogus")


if __name__ == "__main__":
    unittest.main()
