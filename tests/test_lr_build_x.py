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
from unittest.mock import patch

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


def test_build_x_adata_reuses_existing_setup(tmp_path):
    """`build_x_adata` with `reuse_setup=True` reads an already-processed `_adata.h5ad`
    instead of calling `SpaceShip.setup_` -- so this test never trains/builds networks.

    `processed` deliberately has NO 'cell_type' obs column (only 'cell_type_int'), and
    `annot` is left at its default 'cell_type': the relabel/raw_count step must be skipped
    entirely on the reuse path, or this would raise a KeyError.
    """
    processed = _make_adata(seed=5)
    assert "cell_type" not in processed.obs.columns
    setup_dir = tmp_path / "spacetravlr_output"
    (setup_dir / "input_data").mkdir(parents=True)
    processed.write_h5ad(setup_dir / "input_data" / "_adata.h5ad")

    out_path = tmp_path / "lr_x.h5ad"
    result = build_x_adata(
        processed, str(out_path), metabolites=METABOLITES,
        radius=RADIUS, contact_distance=CONTACT, scale_factor=SCALE, setup_dir=setup_dir,
        reuse_setup=True,
    )

    assert result.obsm["x_metab"].shape == (processed.n_obs, 1)
    assert "received_ligands" in result.obsm

    assert out_path.is_file()
    written = ad.read_h5ad(out_path)
    assert "x_metab" in written.obsm
    assert "received_ligands" in written.obsm
    assert written.uns["x_metab_modulators"] == ["metab@M"]


def test_build_x_adata_fresh_setup_threads_args_to_spaceship(tmp_path):
    """The reuse_setup=False (fresh setup_) path threads its args to `SpaceShip` correctly
    -- constructor `genes=`, and `setup_(overwrite=True, run_commot=...)` -- with a fake
    SpaceShip so this never runs the real (expensive) setup_ pipeline. Also proves Fix 1:
    the relabel/raw_count mutation DOES happen on this path (the fake asserts on it)."""
    calls = {}

    class _FakeSpaceShip:
        def __init__(self, name, outdir, genes=None):
            calls["init"] = dict(name=name, outdir=outdir, genes=genes)
            self.adata = None

        def setup_(self, adata, overwrite=False, run_commot=False):
            calls["setup_"] = dict(
                overwrite=overwrite, run_commot=run_commot,
                had_raw_count="raw_count" in adata.layers,
                cell_type=list(adata.obs["cell_type"].unique()),
            )
            self.adata = _make_adata(seed=9)

    raw = _make_adata(seed=8)
    raw.obs["annotation"] = pd.Categorical(["ctA"] * raw.n_obs)
    setup_dir = tmp_path / "spacetravlr_output"
    out_path = tmp_path / "lr_x.h5ad"

    with patch("SpaceTravLR.spaceship.SpaceShip", _FakeSpaceShip):
        result = build_x_adata(
            raw, str(out_path), annot="annotation", metabolites=METABOLITES,
            focus_genes=["G"], radius=RADIUS, contact_distance=CONTACT, scale_factor=SCALE,
            run_commot=True, setup_dir=setup_dir, reuse_setup=False,
        )

    assert calls["init"] == dict(name="lr_setup", outdir=str(setup_dir), genes=["G"])
    assert calls["setup_"]["overwrite"] is True
    assert calls["setup_"]["run_commot"] is True
    assert calls["setup_"]["had_raw_count"] is True
    assert calls["setup_"]["cell_type"] == ["ctA"]

    assert "x_metab" in result.obsm
    assert "received_ligands" in result.obsm
    assert out_path.is_file()


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
