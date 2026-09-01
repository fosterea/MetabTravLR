"""Tier-1 tests for `metab_processing/SpaceTravLR/beta_analysis.compute_metab_x` /
`metab_x_to_adata` -- the metabolite communication-score (`x`) recovery used to build
`beta * x`.

The ground truth is the estimator itself: `SpatialCellularProgramsEstimator.init_data()`
writes the design-matrix metab columns to `adata.uns['metabolite_interactions']`. These tests
assert `compute_metab_x` reproduces that EXACTLY when no transporter pair touches the target
gene (the only case where the two are defined to agree; `compute_metab_x` skips the per-target
self-exclusion by design). Requires torch (via parallel_estimators); runs in the model env, not
the pure-pandas Tier-0 loop.
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

from SpaceTravLR.models.parallel_estimators import SpatialCellularProgramsEstimator
from metab_processing.SpaceTravLR.beta_analysis import compute_metab_x, metab_x_to_adata

RADIUS, CONTACT, SCALE = 100, 30, 100


def _make_adata(genes, target, n=8, seed=0):
    """Tiny AnnData with an `imputed_count` layer, `spatial` obsm and `cell_type_int` obs
    (mirrors tests/test_metab_group.py::_make_adata). `genes` must NOT include `target`."""
    rng = np.random.default_rng(seed)
    all_genes = list(genes) + [target]
    X = rng.random((n, len(all_genes))).astype(np.float32)
    a = ad.AnnData(X=X)
    a.var_names = all_genes
    a.obs_names = [f"c{i}" for i in range(n)]
    a.obs["cell_type_int"] = pd.Categorical(rng.integers(0, 2, n))
    a.obsm["spatial"] = rng.uniform(0, 500, size=(n, 2))
    a.layers["imputed_count"] = X.copy()
    return a


def _ground_truth(adata, target, metabolites):
    """`adata.uns['metabolite_interactions']` as built by the real estimator, on a COPY so the
    diffusion it caches doesn't leak into the adata under test."""
    a = adata.copy()
    est = SpatialCellularProgramsEstimator(
        adata=a, target_gene=target, cluster_annot="cell_type_int",
        layer="imputed_count", radius=RADIUS, contact_distance=CONTACT,
        regulators=[], use_ligands=False, scale_factor=SCALE,
        metabolites=metabolites,
    )
    est.init_data()
    return a.uns["metabolite_interactions"]


def test_matches_estimator_real_diffusion():
    """compute_metab_x == the estimator's metabolite_interactions, through the REAL
    init_received_ligands diffusion path (no pre-seeding), for a target-independent metabolite."""
    genes = ["A", "B"]
    target = "T"
    metabolites = {"M": [("A", "B"), ("B", "A")]}
    adata = _make_adata(genes, target, n=8, seed=3)

    truth = _ground_truth(adata, target, metabolites)
    x = compute_metab_x(adata, metabolites, RADIUS, CONTACT, SCALE, "imputed_count")

    assert list(x.columns) == ["metab@M"]
    assert list(x.index) == list(adata.obs_names)
    np.testing.assert_allclose(x["metab@M"].values, truth["metab@M"].values)


def test_matches_estimator_multiple_metabolites():
    """Two metabolites, one multi-pair, sharing an export gene -- per-metabolite sums match."""
    genes = ["A", "B", "C"]
    target = "T"
    metabolites = {"M1": [("A", "B")], "M2": [("A", "C"), ("B", "B")]}
    adata = _make_adata(genes, target, n=6, seed=4)

    truth = _ground_truth(adata, target, metabolites)
    x = compute_metab_x(adata, metabolites, RADIUS, CONTACT, SCALE, "imputed_count")

    assert list(x.columns) == ["metab@M1", "metab@M2"]
    np.testing.assert_allclose(x["metab@M1"].values, truth["metab@M1"].values)
    np.testing.assert_allclose(x["metab@M2"].values, truth["metab@M2"].values)


def test_does_not_leave_diffusion_frames_in_uns():
    """The big cells x ligands frames init_received_ligands writes are cleaned up when they
    weren't there before (so a saved h5ad doesn't balloon)."""
    adata = _make_adata(["A", "B"], "T", n=8, seed=5)
    assert "received_ligands_tfl" not in adata.uns
    compute_metab_x(adata, {"M": [("A", "B")]}, RADIUS, CONTACT, SCALE, "imputed_count")
    assert "received_ligands_tfl" not in adata.uns
    assert "received_ligands" not in adata.uns


def test_drops_pairs_with_missing_genes():
    """A pair naming a gene absent from var_names is dropped; the metabolite survives on its
    remaining pairs (matches the estimator's var_names filtering)."""
    genes = ["A", "B"]
    target = "T"
    metabolites = {"M": [("A", "B"), ("A", "Z")]}  # Z not a gene
    adata = _make_adata(genes, target, n=6, seed=6)

    truth = _ground_truth(adata, target, metabolites)  # estimator also drops the Z pair
    x = compute_metab_x(adata, metabolites, RADIUS, CONTACT, SCALE, "imputed_count")
    assert list(x.columns) == ["metab@M"]
    np.testing.assert_allclose(x["metab@M"].values, truth["metab@M"].values)


def test_empty_when_no_valid_metabolites():
    adata = _make_adata(["A", "B"], "T", n=5, seed=7)
    x = compute_metab_x(adata, {"M": [("A", "Z")]}, RADIUS, CONTACT, SCALE, "imputed_count")
    assert x.shape[1] == 0
    assert list(x.index) == list(adata.obs_names)


def test_metab_x_to_adata_attaches_obsm_and_labels():
    adata = _make_adata(["A", "B", "C"], "T", n=6, seed=8)
    metabolites = {"M1": [("A", "B")], "M2": [("A", "C"), ("B", "B")]}

    metab_x_to_adata(adata, metabolites, RADIUS, CONTACT, SCALE, "imputed_count")

    assert adata.uns["x_metab_modulators"] == ["metab@M1", "metab@M2"]
    assert adata.obsm["x_metab"].shape == (6, 2)
    x = compute_metab_x(adata, metabolites, RADIUS, CONTACT, SCALE, "imputed_count")
    np.testing.assert_allclose(adata.obsm["x_metab"], x.to_numpy())


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
