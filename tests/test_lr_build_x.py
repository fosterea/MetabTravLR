"""Tier-1 tests for `metab_processing/LinearRegression/build_x.py`.

`store_design_matrices` builds, per focus gene, the REAL
`SpatialCellularProgramsEstimator` and calls `init_data()` -- exactly what training does
(`oracles.py::SpaceTravLR.run`) -- then stores `train_df` minus the target column. So the
ground truth here is the estimator itself, mirroring `tests/test_metab_group.py`'s style.
Requires torch (via parallel_estimators); runs in the model env, not the pure-pandas Tier-0
loop.
"""
import json
import os
import pickle
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import anndata as ad
import pytest
from unittest.mock import patch

from SpaceTravLR.tools.network import RegulatoryFactory
from SpaceTravLR.models.parallel_estimators import SpatialCellularProgramsEstimator
import SpaceTravLR.models.parallel_estimators as pe
from metab_processing.SpaceTravLR.beta_analysis import _group
from metab_processing.LinearRegression import build_x as bx
from metab_processing.LinearRegression.build_x import (
    store_design_matrices, build_x_adata, _CLEANUP_OBSM, _CLEANUP_UNS,
)

RADIUS, CONTACT, SCALE = 100, 30, 100

# TGFB1/TGFBR1/TGFBR2 are real CellChat human L-R genes (as in the old test_lr_build_x.py)
# -- non-trivial for the L-R group. A/B are a plain metabolite transporter pair. TF1 is a
# plain TF wired (via a hand-built grn) to regulate the target, and also wired (via a
# hand-built tflinks) as the top NicheNet regulator for TGFB1, so the L-TF group is
# non-trivial too. This lets one target gene exercise all four modulator groups at once.
MOD_GENES = ["TF1", "TGFB1", "TGFBR1", "TGFBR2", "A", "B"]
TARGET = "T"
METABOLITES = {"M": [("A", "B"), ("B", "A")]}


def _make_adata(mod_genes, targets, n=30, seed=0):
    """Tiny PROCESSED-shaped AnnData: `imputed_count` layer, `spatial` obsm, `cell_type_int`
    obs with 2 clusters (mirrors tests/test_metab_group.py::_make_adata)."""
    rng = np.random.default_rng(seed)
    all_genes = list(mod_genes) + list(targets)
    X = rng.random((n, len(all_genes))).astype(np.float32)
    a = ad.AnnData(X=X)
    a.var_names = all_genes
    a.obs_names = [f"c{i}" for i in range(n)]
    a.obs["cell_type"] = pd.Categorical(["ctA" if i % 2 == 0 else "ctB" for i in range(n)])
    a.obs["cell_type_int"] = pd.Categorical([i % 2 for i in range(n)])
    a.obsm["spatial"] = rng.uniform(0, 500, size=(n, 2))
    a.layers["imputed_count"] = X.copy()
    return a


def _links_dict():
    """A minimal CellOracle-links-shaped dict: TF1 -> T in every cluster."""
    df = pd.DataFrame({"source": ["TF1"], "target": ["T"], "coef_mean": [0.5], "p": [0.01]})
    return {0: df.copy(), 1: df.copy()}


def _build_networks():
    """A real (not mocked) RegulatoryFactory + a minimal tflinks frame: TF1 is the top
    NicheNet regulator of TGFB1, so L-TF produces 'TGFB1#TF1'."""
    grn = RegulatoryFactory(links=_links_dict(), annot="cell_type_int")
    tflinks = pd.DataFrame({"TGFB1": [0.5]}, index=["TF1"])
    return grn, tflinks


def _est_kwargs(**overrides):
    kwargs = dict(radius=RADIUS, contact_distance=CONTACT, scale_factor=SCALE,
                  tf_ligand_cutoff=0.01, receptor_thresh=0.01, layer="imputed_count",
                  cluster_annot="cell_type_int")
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# store_design_matrices
# ---------------------------------------------------------------------------

def test_parity_all_four_groups_match_estimator_directly():
    """Ground truth = an independently-built estimator's own train_df."""
    adata = _make_adata(MOD_GENES, [TARGET], n=40, seed=1)
    grn, tflinks = _build_networks()
    kwargs = _est_kwargs(metabolites=METABOLITES)

    truth_adata = adata.copy()
    truth_est = SpatialCellularProgramsEstimator(
        adata=truth_adata, target_gene=TARGET, grn=grn, tflinks=tflinks, **kwargs)
    truth_est.init_data()
    truth_X = truth_est.train_df.drop(columns=[TARGET])

    result = store_design_matrices(adata, grn, tflinks, [TARGET], **kwargs)

    assert result.uns["x_genes"] == [TARGET]
    assert list(result.uns[f"x_{TARGET}_cols"]) == list(truth_X.columns)
    np.testing.assert_allclose(result.obsm[f"x_{TARGET}"], truth_X.to_numpy())

    # Sanity-check the fixture itself actually exercises all four groups (guards against a
    # silently-degraded fixture making this parity test vacuous).
    groups = {_group(c) for c in truth_X.columns}
    assert groups == {"tf", "lr", "ltf", "metab"}


def test_stale_diffusion_cache_is_cleared_before_rebuild():
    """A pre-existing `received_ligands_tfl` (e.g. from a `run_commot=True` setup, cached at
    a different radius and WITHOUT the metabolite export genes) must not be trusted: if kept,
    init_data's guard (parallel_estimators.py:977) would skip diffusion and reuse it, raising
    KeyError on the metab_exports lookup. store_design_matrices must clear it up front so
    init_data rebuilds fresh -- matches beta_analysis.compute_metab_x's same guard."""
    adata = _make_adata(MOD_GENES, [TARGET], n=40, seed=12)
    grn, tflinks = _build_networks()
    kwargs = _est_kwargs(metabolites=METABOLITES)

    truth_adata = adata.copy()  # a clean copy (no stale cache) is the ground truth

    # Stale cache: present, but missing the metabolite export genes A/B entirely.
    stale = pd.DataFrame({"TGFB1": np.zeros(adata.n_obs)}, index=adata.obs_names)
    adata.uns["received_ligands_tfl"] = stale

    truth_est = SpatialCellularProgramsEstimator(
        adata=truth_adata, target_gene=TARGET, grn=grn, tflinks=tflinks, **kwargs)
    truth_est.init_data()
    truth_X = truth_est.train_df.drop(columns=[TARGET])

    result = store_design_matrices(adata, grn, tflinks, [TARGET], **kwargs)  # must not KeyError

    assert list(result.uns[f"x_{TARGET}_cols"]) == list(truth_X.columns)
    np.testing.assert_allclose(result.obsm[f"x_{TARGET}"], truth_X.to_numpy())


def test_multiple_genes_each_get_own_matrix():
    adata = _make_adata(MOD_GENES, [TARGET, "C2"], n=30, seed=2)
    grn, tflinks = _build_networks()
    kwargs = _est_kwargs(metabolites=METABOLITES)

    with patch.object(pe, "init_received_ligands", wraps=pe.init_received_ligands) as spy:
        result = store_design_matrices(adata, grn, tflinks, [TARGET, "C2"], **kwargs)
        # Diffusion cache built once (first gene) and reused by the second -- init_data
        # only calls init_received_ligands when neither uns cache key is present yet.
        assert spy.call_count == 1

    assert result.uns["x_genes"] == [TARGET, "C2"]
    assert result.obsm[f"x_{TARGET}"].shape[0] == adata.n_obs
    assert result.obsm[f"x_{TARGET}"].shape[1] == len(result.uns[f"x_{TARGET}_cols"])
    # C2 has no TF regulator (grn only wires TF1 -> T) and no L-TF (L-TF requires TF1 to be
    # C2's own regulator too) -- but L-R is not target-specific (only self-exclusion applies)
    # and metabolites apply uniformly, so C2 still gets the shared L-R + metab columns.
    assert result.uns["x_C2_cols"] == ["TGFB1$TGFBR1", "TGFB1$TGFBR2", "metab@M"]
    assert result.obsm["x_C2"].shape == (adata.n_obs, 3)


def test_orphan_gene_skipped_and_warned():
    """A gene with no TF (grn only wires TF1 -> T, not ORPH), no L-R/L-TF (receptor_thresh
    set above every receptor's expression so init_ligands_and_receptors' OWN lr/tfl end up
    empty -- while the global diffusion candidate set init_received_ligands computes is
    unaffected by receptor_thresh and stays non-empty via TGFB1/TGFBR1/TGFBR2, so the
    diffusion itself still runs), and no metab (metabolites=None) -> 0-column X -> skipped
    + warned, not stored in uns['x_genes']."""
    adata = _make_adata(MOD_GENES, ["ORPH"], n=30, seed=5)
    grn, tflinks = _build_networks()

    with pytest.warns(UserWarning, match="no modulators"):
        result = store_design_matrices(adata, grn, tflinks, ["ORPH"],
                                       **_est_kwargs(metabolites=None, receptor_thresh=999))

    assert result.uns["x_genes"] == []
    assert "x_ORPH" not in result.obsm
    assert "x_ORPH_cols" not in result.uns


def test_missing_gene_skipped_and_warned():
    adata = _make_adata(MOD_GENES, [TARGET], n=20, seed=9)
    grn, tflinks = _build_networks()

    with pytest.warns(UserWarning, match="not in adata.var_names"):
        result = store_design_matrices(adata, grn, tflinks, [TARGET, "NOPE"],
                                       **_est_kwargs(metabolites=METABOLITES))

    assert "NOPE" not in result.uns["x_genes"]
    assert TARGET in result.uns["x_genes"]


def test_cleanup_pop_list_removes_diffusion_and_spatial_artifacts():
    """Mirrors build_x_adata step 6: confirms the artifacts it pops actually exist after
    store_design_matrices (guards a stale pop-list), and that popping them leaves the stored
    design matrices intact."""
    adata = _make_adata(MOD_GENES, [TARGET], n=30, seed=6)
    grn, tflinks = _build_networks()
    store_design_matrices(adata, grn, tflinks, [TARGET], **_est_kwargs(metabolites=METABOLITES))

    for key in _CLEANUP_OBSM:
        assert key in adata.obsm, f"expected {key!r} in obsm before cleanup"
    for key in _CLEANUP_UNS:
        assert key in adata.uns, f"expected {key!r} in uns before cleanup"

    for key in _CLEANUP_OBSM:
        adata.obsm.pop(key, None)
    for key in _CLEANUP_UNS:
        adata.uns.pop(key, None)

    for key in _CLEANUP_OBSM:
        assert key not in adata.obsm
    for key in _CLEANUP_UNS:
        assert key not in adata.uns
    assert f"x_{TARGET}" in adata.obsm  # the actual payload survives cleanup


# ---------------------------------------------------------------------------
# build_x_adata
# ---------------------------------------------------------------------------

def _write_setup_dir(setup_dir, proc_adata, tflinks):
    (setup_dir / "input_data").mkdir(parents=True)
    (setup_dir / "betadata").mkdir(parents=True)
    proc_adata.write_h5ad(setup_dir / "input_data" / "_adata.h5ad")
    with open(setup_dir / "input_data" / "celloracle_links.pkl", "wb") as f:
        pickle.dump(_links_dict(), f)
    tflinks.to_parquet(setup_dir / "input_data" / "tflinks.parquet")


def test_build_x_adata_reuses_existing_setup_and_writes_one_file(tmp_path):
    adata = _make_adata(MOD_GENES, [TARGET], n=30, seed=7)
    _, tflinks = _build_networks()
    setup_dir = tmp_path / "spacetravlr_output"
    _write_setup_dir(setup_dir, adata, tflinks)

    out_path = tmp_path / "x_adata.h5ad"
    result = build_x_adata(
        adata, str(out_path), focus_genes=[TARGET], metabolites=METABOLITES,
        setup_dir=setup_dir, **_est_kwargs())

    assert out_path.is_file()
    assert [p.name for p in tmp_path.iterdir() if p.is_file()] == ["x_adata.h5ad"]

    written = ad.read_h5ad(out_path)
    assert TARGET in written.uns["x_genes"]
    assert f"x_{TARGET}" in written.obsm
    for key in _CLEANUP_OBSM:
        assert key not in written.obsm
    for key in _CLEANUP_UNS:
        assert key not in written.uns
    np.testing.assert_allclose(written.obsm[f"x_{TARGET}"], result.obsm[f"x_{TARGET}"])


def test_build_x_adata_run_params_json_overrides_args(tmp_path):
    """run_params.json, when present, is authoritative over the function's default args --
    except receptor_thresh, which run_params.json never persists (see _RUN_PARAM_OVERRIDES'
    comment), so it always stays the caller's default. `layer`/`annot` in the fixture
    deliberately differ from the call's defaults (both point at real, valid alternatives)
    so a broken key-mapping for either would actually be caught here."""
    adata = _make_adata(MOD_GENES, [TARGET], n=20, seed=11)
    adata.layers["imputed_count_alt"] = adata.layers["imputed_count"].copy()
    adata.obs["cell_type_int_alt"] = adata.obs["cell_type_int"]
    _, tflinks = _build_networks()
    setup_dir = tmp_path / "spacetravlr_output"
    _write_setup_dir(setup_dir, adata, tflinks)
    run_params = {
        "radius": 999, "contact_distance": 77, "scale_factor": 5,
        "tf_ligand_cutoff": 0.02, "layer": "imputed_count_alt", "annot": "cell_type_int_alt",
    }
    (setup_dir / "betadata" / "run_params.json").write_text(json.dumps(run_params))

    captured = {}
    real = bx.store_design_matrices

    def _spy(*args, **kw):
        captured.update(kw)
        return real(*args, **kw)

    out_path = tmp_path / "x_adata.h5ad"
    with patch.object(bx, "store_design_matrices", side_effect=_spy):
        build_x_adata(
            adata, str(out_path), focus_genes=[TARGET], metabolites=METABOLITES,
            setup_dir=setup_dir, radius=100, contact_distance=30, scale_factor=100,
            tf_ligand_cutoff=0.01, receptor_thresh=0.42, layer="imputed_count",
            cluster_annot="cell_type_int")

    assert captured["radius"] == 999
    assert captured["contact_distance"] == 77
    assert captured["scale_factor"] == 5
    assert captured["tf_ligand_cutoff"] == 0.02
    assert captured["layer"] == "imputed_count_alt"
    assert captured["cluster_annot"] == "cell_type_int_alt"
    # receptor_thresh has no run_params.json key -> always the caller's default, unchanged.
    assert captured["receptor_thresh"] == 0.42


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
