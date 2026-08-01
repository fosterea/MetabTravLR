"""Tier-0 tests for `metab_processing/SpaceTravLR/beta_analysis.py` — the read-out side of
MetabTravLR. Tiny hand-built betadata parquets with KNOWN beta values, known-answer asserts.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from metab_processing.SpaceTravLR.beta_analysis import (
    betas_to_adata,
    tier_means,
    write_gene_pairs,
    write_histograms,
)

CELLS = [f"c{i}" for i in range(6)]
BETAS = {
    "beta_ABCA1@ABCA1": [1.0, 2, 3, 4, 5, 6],     # metab
    "beta_ATP7A@ATP7B": [0.0, 0, 0, 1, 1, 1],     # metab, heterotypic
    "beta_ATP7B@ATP7A": [-1.0, -1, -1, 0, 0, 0],  # ... and its reverse orientation
    "beta_IL2$IL2RA": [0.5] * 6,                  # lr
    "beta_IL2#STAT5A": [0.1] * 6,                 # ltf
    "beta_STAT1": [2.0] * 6,                      # tf
}


class FakeAnnData:
    """Just the three attributes `betas_to_adata` touches."""

    def __init__(self, obs_names):
        self.obs_names = pd.Index(obs_names)
        self.obsm = {}
        self.uns = {}


class BetaAnalysisTests(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.betadata_dir = self.tmp / "betadata"
        self.betadata_dir.mkdir()
        for gene in ("CD4", "CD3E"):
            pd.DataFrame(BETAS, index=CELLS).to_parquet(
                self.betadata_dir / f"{gene}_betadata.parquet")

        self.obs = pd.DataFrame(
            {"Tier1": pd.Categorical(["T Cell"] * 3 + ["other"] * 3),
             "Tier2": ["a", "a", "b", "b", "c", "c"]},
            index=CELLS,
        )
        self.outdir = self.tmp / "metabtravlr_outputs"

    def test_tier_means_known_values(self):
        means = tier_means(self.betadata_dir, self.obs, "Tier1", genes=["CD4"])
        row = means[(means.modulator == "ABCA1@ABCA1") & (means.cell_type == "T Cell")].iloc[0]
        # cells c0..c2 are 'T Cell': mean([1,2,3]) = 2, sample std = 1
        self.assertEqual((row["mean"], row["std"], row["n"]), (2.0, 1.0, 3))
        self.assertEqual(len(means), len(BETAS) * 2)  # all modulators x 2 cell types

    def test_gene_pairs_csv(self):
        write_gene_pairs(self.betadata_dir, self.obs, ["Tier1", "Tier2"], self.outdir)

        df = pd.read_csv(self.outdir / "Tier1" / "gene_pairs.csv")
        self.assertEqual(list(df.columns),
                         ["gene", "export", "import", "pair", "cell_type", "mean", "std", "n"])
        # metabolite pairs only, both orientations kept as their own rows
        self.assertEqual(set(df.pair), {"ABCA1@ABCA1", "ATP7A@ATP7B", "ATP7B@ATP7A"})
        self.assertEqual(len(df), 2 * 3 * 2)  # genes x pairs x cell types

        reverse = df[(df.gene == "CD4") & (df.pair == "ATP7B@ATP7A") & (df.cell_type == "T Cell")]
        self.assertEqual((reverse.iloc[0]["export"], reverse.iloc[0]["import"]), ("ATP7B", "ATP7A"))
        self.assertEqual(reverse.iloc[0]["mean"], -1.0)

        # a second tier is a second folder, grouped by that tier's labels
        tier2 = pd.read_csv(self.outdir / "Tier2" / "gene_pairs.csv")
        self.assertEqual(sorted(tier2.cell_type.unique()), ["a", "b", "c"])

    def test_write_all_groups(self):
        write_gene_pairs(self.betadata_dir, self.obs, ["Tier1"], self.outdir)
        tier_dir = self.outdir / "Tier1"

        lr = pd.read_csv(tier_dir / "ligand_receptor.csv")
        self.assertEqual(list(lr.columns),
                         ["gene", "ligand", "receptor", "pair", "cell_type", "mean", "std", "n"])
        self.assertEqual(set(lr.pair), {"IL2$IL2RA"})
        self.assertEqual((lr.iloc[0]["ligand"], lr.iloc[0]["receptor"]), ("IL2", "IL2RA"))
        self.assertEqual(len(lr), 2 * 1 * 2)  # genes x lr pairs x cell types
        self.assertTrue((lr["mean"] == 0.5).all())

        ltf = pd.read_csv(tier_dir / "ligand_tf.csv")
        self.assertEqual(list(ltf.columns),
                         ["gene", "ligand", "tf", "pair", "cell_type", "mean", "std", "n"])
        self.assertEqual(set(ltf.pair), {"IL2#STAT5A"})
        self.assertEqual((ltf.iloc[0]["ligand"], ltf.iloc[0]["tf"]), ("IL2", "STAT5A"))
        self.assertTrue((ltf["mean"] == 0.1).all())

        tf = pd.read_csv(tier_dir / "transcription_factor.csv")
        self.assertEqual(list(tf.columns), ["gene", "tf", "cell_type", "mean", "std", "n"])
        self.assertEqual(set(tf.tf), {"STAT1"})
        self.assertTrue((tf["mean"] == 2.0).all())

    def test_write_all_groups_off(self):
        write_gene_pairs(self.betadata_dir, self.obs, ["Tier1"], self.outdir,
                         write_all_groups=False)
        tier_dir = self.outdir / "Tier1"
        self.assertTrue((tier_dir / "gene_pairs.csv").exists())
        for extra in ("ligand_receptor.csv", "ligand_tf.csv", "transcription_factor.csv"):
            self.assertFalse((tier_dir / extra).exists())

    def test_histograms(self):
        write_histograms(self.betadata_dir, self.obs, ["Tier1"], self.outdir, bins=5)

        hist = pd.read_csv(self.outdir / "Tier1" / "histograms.csv")
        self.assertEqual(set(hist.group), {"metab", "lr", "ltf", "tf"})
        self.assertEqual(list(hist.columns), ["group", "left", "right", "count"])
        # every (gene, modulator, cell type) mean lands in exactly one bin of its group
        self.assertEqual(hist[hist.group == "metab"]["count"].sum(), 2 * 3 * 2)
        self.assertEqual(hist[hist.group == "tf"]["count"].sum(), 2 * 1 * 2)
        self.assertEqual(len(hist[hist.group == "lr"]), 5)

    def test_betas_to_adata(self):
        adata = FakeAnnData(CELLS + ["untrained"])
        betas_to_adata(adata, self.betadata_dir, genes=["CD4"])

        self.assertEqual(list(adata.obsm), ["beta_CD4"])
        self.assertEqual(adata.obsm["beta_CD4"].shape, (7, 3))  # metab columns only
        self.assertEqual(adata.uns["beta_modulators"]["CD4"],
                         ["ABCA1@ABCA1", "ATP7A@ATP7B", "ATP7B@ATP7A"])
        # betas stay tied to their cell; a cell the gene wasn't fit on is NaN
        np.testing.assert_array_equal(adata.obsm["beta_CD4"][:6, 0], [1, 2, 3, 4, 5, 6])
        self.assertTrue(np.isnan(adata.obsm["beta_CD4"][-1]).all())


if __name__ == "__main__":
    unittest.main()
