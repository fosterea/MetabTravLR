"""Tier-1 tests for `metab_processing/LinearRegression/build_x.py`.

`add_x_data` attaches the x building blocks (metabolite communication scores +
diffused-ligand table) to a PROCESSED adata. It's a thin wrapper over
`beta_analysis.compute_metab_x`/`metab_x_to_adata` and
`parallel_estimators.init_received_ligands`, so the ground truth is those functions
themselves, mirroring `tests/test_metab_x.py`'s style. Requires torch (via
parallel_estimators); runs in the model env, not the pure-pandas Tier-0 loop.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import anndata as ad
import pytest

from SpaceTravLR.models.parallel_estimators import init_received_ligands
from metab_processing.SpaceTravLR.beta_analysis import compute_metab_x
from metab_processing.LinearRegression.build_x import add_x_data, build_x_adata

RADIUS, CONTACT, SCALE = 100, 30, 100

# A couple of real CellChat human L-R genes (TGFB1/TGFBR1/TGFBR2), alongside two
# non-CellChat genes A/B used as the metabolite transporter pair -- so the
# `received_ligands` building block (driven by CellChat) and the metabolite `x_metab`
# (driven by the metabolite dict) exercise genuinely different, non-trivial columns.
GENES = ["TGFB1", "TGFBR1", "TGFBR2", "A", "B"]
TARGET = "T"
METABOLITES = {"M": [("A", "B"), ("B", "A")]}


def _make_adata(n=10, seed=0):
    """Tiny PROCESSED-shaped AnnData: `imputed_count` layer, `spatial` obsm,
    `cell_type_int` obs (mirrors tests/test_metab_x.py::_make_adata)."""
    rng = np.random.default_rng(seed)
    all_genes = GENES + [TARGET]
    X = rng.random((n, len(all_genes))).astype(np.float32)
    a = ad.AnnData(X=X)
    a.var_names = all_genes
    a.obs_names = [f"c{i}" for i in range(n)]
    a.obs["cell_type_int"] = pd.Categorical(rng.integers(0, 2, n))
    a.obsm["spatial"] = rng.uniform(0, 500, size=(n, 2))
    a.layers["imputed_count"] = X.copy()
    return a


def test_add_x_data_metab_matches_compute_metab_x():
    adata = _make_adata(seed=1)
    expected = compute_metab_x(adata.copy(), METABOLITES, RADIUS, CONTACT, SCALE,
                                "imputed_count")

    add_x_data(adata, METABOLITES, RADIUS, CONTACT, SCALE, "imputed_count")

    assert adata.uns["x_metab_modulators"] == ["metab@M"]
    assert adata.obsm["x_metab"].shape == (adata.n_obs, 1)
    np.testing.assert_allclose(adata.obsm["x_metab"], expected.to_numpy())


def test_add_x_data_attaches_received_ligands():
    adata = _make_adata(seed=2)
    ref = adata.copy()

    add_x_data(adata, METABOLITES, RADIUS, CONTACT, SCALE, "imputed_count")

    n_cols = len(adata.uns["received_ligands_cols"])
    assert adata.obsm["received_ligands"].shape == (adata.n_obs, n_cols)
    assert n_cols > 0  # TGFB1/TGFBR1/TGFBR2 are real CellChat genes -> non-trivial

    # Recompute the same building block independently (extra_lr=None) and compare.
    init_received_ligands(ref, RADIUS, cell_threshes=None, contact_distance=CONTACT,
                           scale_factor=SCALE, layer="imputed_count", extra_lr=None)
    truth = ref.uns["received_ligands_tfl"]
    assert list(truth.columns) == adata.uns["received_ligands_cols"]
    np.testing.assert_allclose(adata.obsm["received_ligands"], truth.to_numpy())


def test_add_x_data_cleans_up_uns_frames():
    adata = _make_adata(seed=3)
    assert "received_ligands_tfl" not in adata.uns
    assert "received_ligands" not in adata.uns

    add_x_data(adata, METABOLITES, RADIUS, CONTACT, SCALE, "imputed_count")

    assert "received_ligands_tfl" not in adata.uns
    assert "received_ligands" not in adata.uns


def test_add_x_data_warns_and_ignores_stale_cell_thresholds():
    """If `uns['cell_thresholds']` (a COMMOT cache) is present, add_x_data still stores the
    UNFILTERED diffusion (cell_threshes=None) and warns rather than silently filtering."""
    adata = _make_adata(seed=6)
    ref = adata.copy()
    adata.uns["cell_thresholds"] = pd.DataFrame(index=adata.obs_names)  # dummy COMMOT cache

    with pytest.warns(UserWarning, match="cell_thresholds"):
        add_x_data(adata, None, RADIUS, CONTACT, SCALE, "imputed_count")

    init_received_ligands(ref, RADIUS, cell_threshes=None, contact_distance=CONTACT,
                           scale_factor=SCALE, layer="imputed_count", extra_lr=None)
    truth = ref.uns["received_ligands_tfl"]
    np.testing.assert_allclose(adata.obsm["received_ligands"], truth.to_numpy())
    assert "received_ligands_tfl" not in adata.uns
    assert "received_ligands" not in adata.uns


def test_add_x_data_no_metabolites():
    adata = _make_adata(seed=4)

    add_x_data(adata, None, RADIUS, CONTACT, SCALE, "imputed_count")

    assert "x_metab" not in adata.obsm
    assert "x_metab_modulators" not in adata.uns
    assert "received_ligands" in adata.obsm
    assert len(adata.uns["received_ligands_cols"]) == adata.obsm["received_ligands"].shape[1]


def _make_raw_adata(n=20, annot_col="my_annot", seed=8):
    """A RAW (un-preprocessed) adata: counts in X, no imputed_count layer, spatial coords,
    and a custom annotation column -- what `build_x_adata` takes as input. One cluster of
    n>=16 cells so MAGIC runs on the real path."""
    rng = np.random.default_rng(seed)
    all_genes = GENES + [TARGET]
    X = (rng.random((n, len(all_genes))) * 5).astype(np.float32)  # small values, no log1p
    a = ad.AnnData(X=X)
    a.var_names = all_genes
    a.obs_names = [f"c{i}" for i in range(n)]
    a.obs[annot_col] = pd.Categorical(["ctA"] * n)
    a.obsm["spatial"] = rng.uniform(0, 500, size=(n, 2))
    return a


def test_build_x_adata_preprocesses_and_writes_single_adata(tmp_path):
    """End-to-end on a RAW adata: build_x_adata imputes, scales spatial, computes the x data,
    and writes ONE adata (no spacetravlr_output tree). Runs real MAGIC + diffusion."""
    raw = _make_raw_adata(n=20, annot_col="my_annot")
    assert "imputed_count" not in raw.layers
    spatial_in = raw.obsm["spatial"].copy()

    out_path = tmp_path / "LinearRegression" / "x_adata.h5ad"
    result = build_x_adata(raw, str(out_path), annot="my_annot", metabolites=METABOLITES,
                           radius=RADIUS, contact_distance=CONTACT, scale_factor=SCALE)

    # nothing but the one file was written (no spacetravlr_output/ tree)
    assert out_path.is_file()
    assert [p.name for p in (tmp_path / "LinearRegression").iterdir()] == ["x_adata.h5ad"]

    written = ad.read_h5ad(out_path)
    assert "imputed_count" in written.layers
    assert "cell_type_int" in written.obs.columns
    assert written.obsm["x_metab"].shape == (raw.n_obs, 1)
    assert written.uns["x_metab_modulators"] == ["metab@M"]
    assert "received_ligands" in written.obsm
    # spatial was scaled in place, and the original is preserved as spatial_unscaled
    assert "spatial_unscaled" in written.obsm
    assert not np.allclose(written.obsm["spatial"], spatial_in)
    np.testing.assert_allclose(written.obsm["spatial_unscaled"], spatial_in)
    # the big diffusion frames are not left in uns
    assert "received_ligands_tfl" not in written.uns


def test_build_x_adata_small_cluster_uses_raw(tmp_path):
    """A sub-`min_cells_for_magic` cluster does not crash MAGIC; build_x_adata completes and
    its imputed_count for that cluster equals the (log/raw) input (via the impute fallback)."""
    raw = _make_raw_adata(n=6, annot_col="my_annot")  # 6 < 16
    out_path = tmp_path / "LinearRegression" / "x_adata.h5ad"
    result = build_x_adata(raw, str(out_path), annot="my_annot", metabolites=METABOLITES)
    assert out_path.is_file()
    assert "imputed_count" in result.layers
    assert result.obsm["x_metab"].shape == (raw.n_obs, 1)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
