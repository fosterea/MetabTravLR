"""Tier-0/1 tests for `metab_processing/LinearRegression/linear_factors.py`.

`linear_factors` builds the SAME LinearRegression factor store (`obsm['x_factors']` +
`uns['x_factors_cols']` + `uns['x_factor_map']` + `uns['x_genes']`, plus
`obsm['x_metab']` + `uns['x_metab_modulators']`) as `build_x.build_factor_block` /
`build_x.add_metabolites`, but WITHOUT importing `SpaceTravLR` (so no torch) --
everything is numpy/pandas/scipy/scikit-learn, with small helpers lifted (copied, cited)
from `src/SpaceTravLR/models/parallel_estimators.py` and `src/SpaceTravLR/tools/
{utils,network}.py`. Fixtures are built entirely by hand here (a tiny synthetic AnnData +
a tiny hand-written base-GRN parquet / CellChat csv / NicheNet tflinks DataFrame) -- this
file never imports SpaceTravLR.
"""
import os
import subprocess
import sys
import tempfile
import unittest
import warnings

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import anndata as ad

from metab_processing.LinearRegression import linear_factors as lf
from metab_processing.LinearRegression.build_x import LAYER, get_gene_factors
from metab_processing.LinearRegression.least_squares import fit_gene_betas

N = 40
RADIUS, CONTACT, SCALE = 100, 30, 1

# Modulator genes:
#   TF1     -- a real TF regulator of T (and of LIGA, for the self-exclusion test).
#   TFZERO  -- a TF regulator of T with ALL-ZERO expression (drop-zero test).
#   LIGA    -- a CellChat ligand, receives TF1 as its NicheNet top ligand-target hit.
#   RECA    -- LIGA's CellChat receptor.
#   LIGB    -- a second CellChat ligand with ALL-ZERO expression (drop-zero test, L-R
#              side: LIGB$RECB should vanish even though RECB itself is nonzero).
#   RECB    -- LIGB's receptor.
#   EXPORT1/IMPORT1 -- a metabolite transporter pair (standalone add_metabolites).
#   T       -- the target/focus gene.
GENES = ["TF1", "TFZERO", "LIGA", "RECA", "LIGB", "RECB", "EXPORT1", "IMPORT1", "T"]


def _make_adata(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((N, len(GENES))).astype(np.float32) + 0.1  # keep strictly positive
    a = ad.AnnData(X=X.copy())
    a.var_names = GENES
    a.obs_names = [f"c{i}" for i in range(N)]
    a.obs["ct"] = pd.Categorical(["A" if i % 2 == 0 else "B" for i in range(N)])
    a.obsm["spatial"] = rng.uniform(0, 500, size=(N, 2))

    zero_mask = np.ones(len(GENES), dtype=bool)
    X[:, GENES.index("TFZERO")] = 0.0
    X[:, GENES.index("LIGB")] = 0.0

    a.layers["raw_count"] = X.copy()
    a.layers["normalized_count"] = X.copy()
    a.X = X.copy()
    return a


def _write_fixture_data_dir(tmpdir, species="human"):
    """Writes `{species}_base_grn.parquet` + `cellchat_{species}.csv` to `tmpdir`,
    matching the real files' schema (verified against the shipped
    `src/SpaceTravLR_data/human_base_grn.parquet` / `cellchat_human.csv`)."""
    base_grn = pd.DataFrame({
        "peak_id": ["p1", "p2", "p3"],
        "gene_short_name": ["T", "LIGA", "RECA"],
        # TF1 regulates both T and LIGA (used by the self-exclusion test); TFZERO
        # regulates T too, but has all-zero EXPRESSION (not GRN presence) -- the
        # all-zero-expression drop happens at the factor-VALUE level (step 7), not here.
        "TF1": [1.0, 1.0, 0.0],
        "TFZERO": [1.0, 0.0, 0.0],
    })
    base_grn.to_parquet(os.path.join(tmpdir, f"{species}_base_grn.parquet"))

    cellchat = pd.DataFrame([
        {"ligand": "LIGA", "receptor": "RECA", "pathway": "X", "signaling": "Secreted Signaling"},
        {"ligand": "LIGB", "receptor": "RECB", "pathway": "X", "signaling": "Secreted Signaling"},
    ])
    cellchat.to_csv(os.path.join(tmpdir, f"cellchat_{species}.csv"), index=False)
    return tmpdir


def _tflinks():
    # Index = TFs, columns = ligands, value = regulatory potential. TF1's top (only)
    # ligand hit is LIGA, well above the default tf_ligand_cutoff (0.01).
    return pd.DataFrame({"LIGA": [0.5], "LIGB": [0.0]}, index=["TF1"])


class NoSpaceTravLRImportTests(unittest.TestCase):
    """Item 1: importing `linear_factors` alone must pull in neither `SpaceTravLR` nor
    `torch`. Run in a FRESH subprocess -- checking `sys.modules` in-process would be
    unreliable inside the full test suite, where sibling test files (e.g.
    `test_lr_build_x.py`) already import SpaceTravLR/torch earlier in the same process.
    """

    def test_fresh_import_has_no_spacetravlr_or_torch(self):
        code = (
            "import sys\n"
            "import metab_processing.LinearRegression.linear_factors\n"
            "bad = [m for m in sys.modules if m == 'SpaceTravLR' or m.startswith('SpaceTravLR.')]\n"
            "assert not bad, f'SpaceTravLR modules imported: {bad}'\n"
            "assert 'torch' not in sys.modules, 'torch imported'\n"
            "print('OK')\n"
        )
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=repo_root,
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("OK", result.stdout)


class BuildLinearFactorStoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = _write_fixture_data_dir(self._tmp.name)
        self.tflinks = _tflinks()
        self.adata = _make_adata()

    def tearDown(self):
        self._tmp.cleanup()

    def _build(self, focus_genes, **kwargs):
        kw = dict(
            annot_col="ct", radius=RADIUS, contact_distance=CONTACT, scale_factor=SCALE,
            receptor_thresh=0.01, tf_ligand_cutoff=0.01, species="human",
            tflinks=self.tflinks, data_dir=self.data_dir,
        )
        kw.update(kwargs)
        return lf.build_linear_factor_store(self.adata, focus_genes, **kw)

    def test_tf_column_equals_raw_expression(self):
        """Item 2: a TF column's values equal `normalized_count[TF]` exactly."""
        self._build(["T"])
        cols = self.adata.uns["x_factors_cols"]
        self.assertIn("TF1", cols)
        idx = cols.index("TF1")
        expected = self.adata[:, "TF1"].to_df(LAYER)["TF1"].to_numpy()
        np.testing.assert_array_equal(self.adata.obsm["x_factors"][:, idx], expected)

    def test_lr_column_matches_direct_diffuse_call(self):
        """Item 3: the `lig$rec` column equals received(lig) * rec_expr, verified by an
        INDEPENDENT direct call to `diffuse` on a masked (single-column) ligand frame."""
        self._build(["T"])
        cols = self.adata.uns["x_factor_map"]["T"]
        self.assertIn("LIGA$RECA", cols)

        xy = lf.compute_scaled_xy(self.adata)
        lig_masked = self.adata[:, ["LIGA"]].to_df(LAYER)
        received = lf.diffuse(xy, lig_masked, RADIUS, SCALE, eps=1e-3)
        rec_expr = self.adata[:, "RECA"].to_df(LAYER)["RECA"].to_numpy()
        expected = received["LIGA"].to_numpy() * rec_expr

        all_cols = self.adata.uns["x_factors_cols"]
        idx = all_cols.index("LIGA$RECA")
        np.testing.assert_allclose(self.adata.obsm["x_factors"][:, idx], expected,
                                   rtol=1e-5, atol=1e-6)

    def test_ltf_column_matches_direct_diffuse_call(self):
        """Item 4: the `lig#tf` column equals received(lig) * tf_expr, same direct-diffuse
        check as the L-R case."""
        self._build(["T"])
        cols = self.adata.uns["x_factor_map"]["T"]
        self.assertIn("LIGA#TF1", cols)

        xy = lf.compute_scaled_xy(self.adata)
        lig_masked = self.adata[:, ["LIGA"]].to_df(LAYER)
        received = lf.diffuse(xy, lig_masked, RADIUS, SCALE, eps=1e-3)
        tf_expr = self.adata[:, "TF1"].to_df(LAYER)["TF1"].to_numpy()
        expected = received["LIGA"].to_numpy() * tf_expr

        all_cols = self.adata.uns["x_factors_cols"]
        idx = all_cols.index("LIGA#TF1")
        np.testing.assert_allclose(self.adata.obsm["x_factors"][:, idx], expected,
                                   rtol=1e-5, atol=1e-6)

    def test_ltf_top5_backfills_when_target_is_a_top_ranked_ligand(self):
        """Regression (reviewer MAJOR 1): when the target gene is itself one of a TF's
        top-5 NicheNet ligands, it must be removed from the ligand pool BEFORE the top-5 is
        taken, so the next-ranked ligand backfills. TF1 ranks LIGA>LIGB>...>LIGF; LIGA is a
        base-GRN target of TF1. For gene LIGA, the pool is LIGB..LIGF -> top-5 = all five,
        so LIGF#TF1 MUST be a modulator of LIGA. For gene T (not a ligand), the global
        top-5 is LIGA..LIGE, so LIGF#TF1 must NOT be in T's map."""
        ligs = ["LIGA", "LIGB", "LIGC", "LIGD", "LIGE", "LIGF"]
        recs = [f"REC{c}" for c in "ABCDEF"]
        genes = ["TF1", "T"] + ligs + recs
        rng = np.random.default_rng(1)
        X = rng.random((N, len(genes))).astype(np.float32) + 0.1
        a = ad.AnnData(X=X.copy()); a.var_names = genes
        a.obs_names = [f"c{i}" for i in range(N)]
        a.obs["ct"] = pd.Categorical(["A" if i % 2 == 0 else "B" for i in range(N)])
        a.obsm["spatial"] = rng.uniform(0, 500, size=(N, 2))
        a.layers["normalized_count"] = X.copy(); a.X = X.copy()

        with tempfile.TemporaryDirectory() as tmp:
            # TF1 regulates BOTH T and LIGA (so TF1 is a regulator of each -> its L-TF pairs apply).
            pd.DataFrame({"peak_id": ["p1", "p2"], "gene_short_name": ["T", "LIGA"],
                          "TF1": [1.0, 1.0]}).to_parquet(os.path.join(tmp, "human_base_grn.parquet"))
            pd.DataFrame([{"ligand": l, "receptor": r, "pathway": "X",
                           "signaling": "Secreted Signaling"} for l, r in zip(ligs, recs)]
                         ).to_csv(os.path.join(tmp, "cellchat_human.csv"), index=False)
            tflinks = pd.DataFrame({l: [v] for l, v in zip(ligs, [0.9, 0.8, 0.7, 0.6, 0.5, 0.4])},
                                   index=["TF1"])
            lf.build_linear_factor_store(
                a, ["T", "LIGA"], annot_col="ct", radius=RADIUS, contact_distance=CONTACT,
                scale_factor=SCALE, species="human", tflinks=tflinks, data_dir=tmp)

        self.assertIn("LIGF#TF1", a.uns["x_factor_map"]["LIGA"])   # backfilled
        self.assertNotIn("LIGA#TF1", a.uns["x_factor_map"]["LIGA"])  # self-excluded
        self.assertNotIn("LIGF#TF1", a.uns["x_factor_map"]["T"])    # 6th, not in global top-5

    def test_ligand_radius_sourced_before_receptor_filter(self):
        """Regression (reviewer MAJOR 2): a ligand's diffusion radius is resolved from the
        var_names-filtered table BEFORE the receptor_thresh filter. LIGX's first pair
        (LIGX$RECX1, Secreted) has a receptor that FAILS receptor_thresh; its surviving pair
        (LIGX$RECX2, Contact) has an expressed receptor. The surviving column must still be
        diffused at the SECRETED radius (first occurrence), not the contact radius."""
        genes = ["T", "LIGX", "RECX1", "RECX2"]
        rng = np.random.default_rng(2)
        X = rng.random((N, len(genes))).astype(np.float32) + 0.1
        X[:, genes.index("RECX1")] = 0.001           # fails receptor_thresh (0.01)
        a = ad.AnnData(X=X.copy()); a.var_names = genes
        a.obs_names = [f"c{i}" for i in range(N)]
        a.obs["ct"] = pd.Categorical(["A" if i % 2 == 0 else "B" for i in range(N)])
        a.obsm["spatial"] = rng.uniform(0, 500, size=(N, 2))
        a.layers["normalized_count"] = X.copy(); a.X = X.copy()

        big_r, small_r = 150, 20
        with tempfile.TemporaryDirectory() as tmp:
            pd.DataFrame({"peak_id": ["p1"], "gene_short_name": ["T"], "LIGX": [0.0]}
                        ).to_parquet(os.path.join(tmp, "human_base_grn.parquet"))
            # RECX1 (Secreted) listed FIRST, RECX2 (Contact) second.
            pd.DataFrame([
                {"ligand": "LIGX", "receptor": "RECX1", "pathway": "X", "signaling": "Secreted Signaling"},
                {"ligand": "LIGX", "receptor": "RECX2", "pathway": "X", "signaling": "Cell-Cell Contact"},
            ]).to_csv(os.path.join(tmp, "cellchat_human.csv"), index=False)
            lf.build_linear_factor_store(
                a, ["T"], annot_col="ct", radius=big_r, contact_distance=small_r,
                scale_factor=SCALE, species="human", tflinks=pd.DataFrame(), data_dir=tmp)

        cols = a.uns["x_factors_cols"]
        self.assertIn("LIGX$RECX2", cols)                  # the surviving pair
        self.assertNotIn("LIGX$RECX1", cols)               # dropped by receptor_thresh
        xy = lf.compute_scaled_xy(a)
        lig_masked = a[:, ["LIGX"]].to_df(LAYER)
        secreted = lf.diffuse(xy, lig_masked, big_r, SCALE, eps=1e-3)["LIGX"].to_numpy()
        contact = lf.diffuse(xy, lig_masked, small_r, SCALE, eps=1e-3)["LIGX"].to_numpy()
        rec2 = a[:, "RECX2"].to_df(LAYER)["RECX2"].to_numpy()
        got = a.obsm["x_factors"][:, cols.index("LIGX$RECX2")]
        np.testing.assert_allclose(got, secreted * rec2, rtol=1e-5, atol=1e-6)   # secreted radius
        self.assertFalse(np.allclose(got, contact * rec2))                       # NOT contact

    def test_self_exclusion(self):
        """Item 5: no column in a gene's own map references that gene -- no `g$*`,
        `*$g`, `g#*`, `*#g`, and the bare-TF columns never equal `g`. Build for TWO focus
        genes at once ('T' and 'LIGA' -- LIGA is itself a CellChat ligand AND a TF1
        target in the base GRN), so LIGA's own map must exclude `LIGA$RECA`,
        `LIGA#TF1`, while T's map (unaffected) keeps both."""
        self._build(["T", "LIGA"])

        liga_cols = self.adata.uns["x_factor_map"]["LIGA"]
        for c in liga_cols:
            self.assertNotEqual(c, "LIGA")
            self.assertFalse(c.startswith("LIGA$"))
            self.assertFalse(c.endswith("$LIGA"))
            self.assertFalse(c.startswith("LIGA#"))
            self.assertFalse(c.endswith("#LIGA"))

        t_cols = self.adata.uns["x_factor_map"]["T"]
        self.assertIn("LIGA$RECA", t_cols)
        self.assertIn("LIGA#TF1", t_cols)

    def test_drop_all_zero_columns(self):
        """Item 6: TFZERO (all-zero TF expression) and LIGB$RECB (all-zero ligand ->
        all-zero received -> all-zero product, even though RECB itself is nonzero)
        produce NO column, and `x_factors_cols` excludes both."""
        self._build(["T"])
        cols = self.adata.uns["x_factors_cols"]
        self.assertNotIn("TFZERO", cols)
        self.assertNotIn("LIGB$RECB", cols)
        for c in self.adata.uns["x_factor_map"]["T"]:
            self.assertNotEqual(c, "TFZERO")
            self.assertNotEqual(c, "LIGB$RECB")
        # sanity: the surviving columns are exactly the non-zero ones we expect.
        self.assertEqual(set(self.adata.uns["x_factor_map"]["T"]),
                         {"TF1", "LIGA$RECA", "LIGA#TF1"})

    def test_missing_focus_gene_warns_and_is_skipped(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._build(["T", "NOT_A_GENE"])
            self.assertTrue(any("NOT_A_GENE" in str(x.message) for x in w))
        self.assertNotIn("NOT_A_GENE", self.adata.uns["x_genes"])
        self.assertIn("T", self.adata.uns["x_genes"])

    def test_store_is_consumable_by_build_x_and_least_squares(self):
        """Item 7: the produced store works unchanged with
        `build_x.get_gene_factors` and `least_squares.fit_gene_betas`."""
        self._build(["T"])
        gf = get_gene_factors(self.adata, "T")
        self.assertEqual(list(gf.columns), self.adata.uns["x_factor_map"]["T"])
        self.assertEqual(len(gf), N)

        df = fit_gene_betas(self.adata, genes=["T"], metabolites=None, method="OLS")
        self.assertFalse(df.empty)
        self.assertEqual(set(df["gene"]), {"T"})
        self.assertEqual(set(df["factor"]), set(self.adata.uns["x_factor_map"]["T"]))
        self.assertTrue(np.isfinite(df["beta"]).all())


class AddMetabolitesTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = _write_fixture_data_dir(self._tmp.name)
        self.adata = _make_adata()

    def tearDown(self):
        self._tmp.cleanup()

    def test_metab_column_equals_summed_received_times_import(self):
        """Item 8: a `metab@<name>` column equals summed received(export) * import,
        verified against an independent direct `diffuse` call."""
        metabolites = {"Glucose": [("EXPORT1", "IMPORT1")]}
        lf.add_metabolites(self.adata, metabolites, radius=RADIUS,
                           contact_distance=CONTACT, scale_factor=SCALE, eps=1e-3,
                           data_dir=self.data_dir)

        self.assertEqual(self.adata.uns["x_metab_modulators"], ["metab@Glucose"])

        xy = lf.compute_scaled_xy(self.adata)
        export_masked = self.adata[:, ["EXPORT1"]].to_df(LAYER)
        received = lf.diffuse(xy, export_masked, RADIUS, SCALE, eps=1e-3)
        import_expr = self.adata[:, "IMPORT1"].to_df(LAYER)["IMPORT1"].to_numpy()
        expected = received["EXPORT1"].to_numpy() * import_expr

        idx = self.adata.uns["x_metab_modulators"].index("metab@Glucose")
        np.testing.assert_allclose(self.adata.obsm["x_metab"][:, idx], expected,
                                   rtol=1e-5, atol=1e-6)

    def test_both_orientations_summed(self):
        """A heterotypic pair's both orientations sum into the same metabolite column
        (D3/D11 convention)."""
        metabolites = {"Glucose": [("EXPORT1", "IMPORT1"), ("IMPORT1", "EXPORT1")]}
        lf.add_metabolites(self.adata, metabolites, radius=RADIUS,
                           contact_distance=CONTACT, scale_factor=SCALE, eps=1e-3,
                           data_dir=self.data_dir)

        xy = lf.compute_scaled_xy(self.adata)
        both = self.adata[:, ["EXPORT1", "IMPORT1"]].to_df(LAYER)
        received = lf.diffuse(xy, both, RADIUS, SCALE, eps=1e-3)
        expected = (received["EXPORT1"].to_numpy() * both["IMPORT1"].to_numpy()
                   + received["IMPORT1"].to_numpy() * both["EXPORT1"].to_numpy())

        idx = self.adata.uns["x_metab_modulators"].index("metab@Glucose")
        np.testing.assert_allclose(self.adata.obsm["x_metab"][:, idx], expected,
                                   rtol=1e-5, atol=1e-6)

    def test_merge_is_idempotent_by_name(self):
        metabolites = {"Glucose": [("EXPORT1", "IMPORT1")]}
        lf.add_metabolites(self.adata, metabolites, radius=RADIUS,
                           contact_distance=CONTACT, scale_factor=SCALE, eps=1e-3,
                           data_dir=self.data_dir)
        before = self.adata.obsm["x_metab"].copy()
        before_cols = list(self.adata.uns["x_metab_modulators"])

        lf.add_metabolites(self.adata, metabolites, radius=RADIUS,
                           contact_distance=CONTACT, scale_factor=SCALE, eps=1e-3,
                           data_dir=self.data_dir)

        np.testing.assert_array_equal(self.adata.obsm["x_metab"], before)
        self.assertEqual(self.adata.uns["x_metab_modulators"], before_cols)

    def test_merge_adds_new_metabolite_without_touching_existing(self):
        lf.add_metabolites(self.adata, {"Glucose": [("EXPORT1", "IMPORT1")]}, radius=RADIUS,
                           contact_distance=CONTACT, scale_factor=SCALE, eps=1e-3,
                           data_dir=self.data_dir)
        glucose_before = self.adata.obsm["x_metab"][:, 0].copy()

        lf.add_metabolites(self.adata, {"Other": [("IMPORT1", "EXPORT1")]}, radius=RADIUS,
                           contact_distance=CONTACT, scale_factor=SCALE, eps=1e-3,
                           data_dir=self.data_dir)

        self.assertEqual(self.adata.uns["x_metab_modulators"], ["metab@Glucose", "metab@Other"])
        np.testing.assert_array_equal(self.adata.obsm["x_metab"][:, 0], glucose_before)

    def test_missing_gene_pair_dropped_and_empty_metabolite_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            lf.add_metabolites(self.adata, {"Bad": [("EXPORT1", "NOT_A_GENE")]},
                               radius=RADIUS, contact_distance=CONTACT, scale_factor=SCALE,
                               eps=1e-3, data_dir=self.data_dir)
            self.assertTrue(any("no metabolite" in str(x.message) for x in w))
        self.assertEqual(self.adata.uns.get("x_metab_modulators"), [])

    def test_metab_x_consumable_by_fit_gene_betas(self):
        """The metabolite block, built standalone (no SpaceTravLR import), is readable
        by `get_gene_factors`/`fit_gene_betas` exactly like the metab block built by
        `build_x.add_metabolites`."""
        lf.add_metabolites(self.adata, {"Glucose": [("EXPORT1", "IMPORT1")]}, radius=RADIUS,
                           contact_distance=CONTACT, scale_factor=SCALE, eps=1e-3,
                           data_dir=self.data_dir)
        self._tmp2 = tempfile.TemporaryDirectory()
        data_dir2 = _write_fixture_data_dir(self._tmp2.name)
        lf.build_linear_factor_store(
            self.adata, ["T"], annot_col="ct", radius=RADIUS, contact_distance=CONTACT,
            scale_factor=SCALE, receptor_thresh=0.01, tf_ligand_cutoff=0.01,
            species="human", tflinks=_tflinks(), data_dir=data_dir2,
        )
        gf = get_gene_factors(self.adata, "T", metabs="all")
        self.assertIn("metab@Glucose", gf.columns)

        df = fit_gene_betas(self.adata, genes=["T"], metabolites="all", method="OLS")
        self.assertIn("metab@Glucose", set(df["factor"]))
        row = df[df["factor"] == "metab@Glucose"].iloc[0]
        self.assertEqual(row["group"], "metab")
        self.assertTrue(np.isfinite(row["beta"]))
        self._tmp2.cleanup()


if __name__ == "__main__":
    unittest.main()
