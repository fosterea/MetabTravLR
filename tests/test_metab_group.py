"""
Tests for CU-1: metabolite transporter pairs as a new, independent modulator
group (D6) in `SpatialCellularProgramsEstimator`.

A metabolite edge = (export gene, import gene). It reuses the L-R
*computation* (received_ligand(export, diffused) x import(local)) but is its
OWN group-lasso group (#5) with a distinct betadata separator `@` (vs `$` for
L-R and `#` for L-TF). The caller supplies both orientations already; the
estimator does not add orientations, it just builds one column per supplied
pair.

Tier 0 tests use `regulators=[]` (bypassing the CellOracle GRN entirely) and
`use_ligands=False` (bypassing CellChat) to stay self-contained and fast --
metab_pairs processing does not depend on either. Tier 1 (real training)
mirrors `tests/test_get_betas_batching.py::_trained_estimator`.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np
import pandas as pd
import anndata as ad
import pytest
from unittest.mock import patch

from SpaceTravLR.tools.network import RegulatoryFactory
import SpaceTravLR.models.parallel_estimators as pe
from SpaceTravLR.models.parallel_estimators import SpatialCellularProgramsEstimator


def _make_adata(genes, target, n=6, seed=0):
    """Minimal AnnData with everything the estimator's ctor asserts on:
    a `spatial` obsm, an `imputed_count` layer, and a `cell_type_int` obs
    column. `genes` should NOT include `target`; it is appended."""
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


def _build_est(genes, target, metab_pairs, regulators=None, **kwargs):
    a = _make_adata(genes, target)
    est = SpatialCellularProgramsEstimator(
        adata=a, target_gene=target, cluster_annot="cell_type_int",
        layer="imputed_count", radius=100, contact_distance=30,
        regulators=[] if regulators is None else regulators,
        use_ligands=False, scale_factor=100,
        metab_pairs=metab_pairs,
        **kwargs,
    )
    return a, est


# ---------------------------------------------------------------------------
# Tier 0: pure-logic / no training
# ---------------------------------------------------------------------------

def test_column_construction_and_ordering():
    """metab_pairs -> 'export@import' columns, appended AFTER regulators
    (and after the -- here empty -- L-R/L-TF/extra groups) in `modulators`."""
    genes = ["R1", "R2", "A", "B", "C"]
    target = "T"
    metab_pairs = [("A", "B"), ("B", "A"), ("C", "C")]  # incl. a homotypic pair
    _, est = _build_est(genes, target, metab_pairs, regulators=["R1", "R2"])

    assert est.metab_pairs == ["A@B", "B@A", "C@C"]
    assert est.modulators == ["R1", "R2", "A@B", "B@A", "C@C"]
    assert est.modulators[-3:] == est.metab_pairs


def test_target_exclusion():
    """A pair touching target_gene is excluded from est.metab_pairs (a gene
    can't predict itself), but IS allowed to remain in the diffusion list
    (target-agnostic by design, so the shared received_ligands_tfl cache
    stays complete across genes)."""
    genes = ["A", "B", "C"]
    target = "T"
    metab_pairs = [("A", "T"), ("B", "C")]
    _, est = _build_est(genes, target, metab_pairs)

    assert est.metab_pairs == ["B@C"]
    assert est.metab_exports == ["B"]
    assert est.metab_imports == ["C"]
    # still present in the diffusion-only list
    assert ("A", "T") in est._diffusion_extra_lr
    assert ("B", "C") in est._diffusion_extra_lr


def test_missing_gene_dropped_not_raised(capsys):
    """A pair referencing a gene not in adata.var_names is dropped (with a
    printed count), not raised."""
    genes = ["A", "B"]
    target = "T"
    metab_pairs = [("A", "B"), ("A", "Z")]  # Z not a gene
    _, est = _build_est(genes, target, metab_pairs)

    assert est.metab_pairs == ["A@B"]
    out = capsys.readouterr().out
    assert "1" in out and "metab_pairs" in out


def test_dedup_exact_duplicate():
    genes = ["A", "B"]
    target = "T"
    metab_pairs = [("A", "B"), ("A", "B")]
    _, est = _build_est(genes, target, metab_pairs)

    assert est.metab_pairs == ["A@B"]


def test_type_validation_non_list_raises():
    genes = ["A", "B"]
    target = "T"
    with pytest.raises(ValueError):
        _build_est(genes, target, metab_pairs={"A": "B"})


def test_type_validation_triple_raises():
    genes = ["A", "B", "C"]
    target = "T"
    with pytest.raises(ValueError):
        _build_est(genes, target, metab_pairs=[("A", "B", "C")])


def test_init_data_metabolite_frame_known_answer():
    """After init_data(), adata.uns['metabolite_interactions'] has EXACTLY
    est.metab_pairs as columns; values equal
    received_ligands_tfl[export] * counts[import] (RAW, not diffused, on the
    import side). We pre-seed received_ligands_tfl with values distinct from
    raw X so the assertion actually distinguishes 'used received/diffused
    export' from 'used raw export'."""
    genes = ["A", "B"]
    target = "T"
    n = 4
    a = _make_adata(genes, target, n=n, seed=1)

    # Raw ('imputed_count') values, known.
    raw_A = np.array([1.0, 2.0, 3.0, 4.0])
    raw_B = np.array([10.0, 20.0, 30.0, 41.0])
    a.X[:, list(a.var_names).index("A")] = raw_A
    a.X[:, list(a.var_names).index("B")] = raw_B
    a.layers["imputed_count"] = a.X.copy()

    # Distinct, hand-built "diffused" received-ligand values for A and B.
    received_A = np.array([100.0, 200.0, 300.0, 400.0])
    received_B = np.array([1000.0, 2000.0, 3000.0, 4001.0])
    a.uns["received_ligands_tfl"] = pd.DataFrame(
        {"A": received_A, "B": received_B}, index=a.obs_names
    )

    metab_pairs = [("A", "B"), ("B", "A")]
    est = SpatialCellularProgramsEstimator(
        adata=a, target_gene=target, cluster_annot="cell_type_int",
        layer="imputed_count", radius=100, contact_distance=30,
        regulators=[], use_ligands=False, scale_factor=100,
        metab_pairs=metab_pairs,
    )

    est.init_data()  # must not raise (exercises the L974 consistency assert)

    frame = a.uns["metabolite_interactions"]
    assert list(frame.columns) == est.metab_pairs == ["A@B", "B@A"]

    expected_AB = received_A * raw_B  # received(export=A) * raw(import=B)
    expected_BA = received_B * raw_A  # received(export=B) * raw(import=A)
    np.testing.assert_array_equal(frame["A@B"].values, expected_AB)
    np.testing.assert_array_equal(frame["B@A"].values, expected_BA)

    # train_df column order: target, regulators, L-R, L-TF, extra, metab
    assert list(est.train_df.columns) == [target, "A@B", "B@A"]


def test_duplicate_gene_flux_values_positional_zip():
    """Guards the positional zip in `metabolite_interactions` against a future
    regression: exports=[A,A,B], imports=[B,C,B] (a repeated export gene AND a
    repeated import gene, incl. one homotypic pair) must produce columns
    ['A@B','A@C','B@B'] with each column's VALUES computed from the correct
    (export, import) pairing, not some other alignment."""
    genes = ["A", "B", "C"]
    target = "T"
    n = 3
    a = _make_adata(genes, target, n=n, seed=4)

    raw_A = np.array([1.0, 2.0, 3.0])
    raw_B = np.array([10.0, 20.0, 30.0])
    raw_C = np.array([100.0, 200.0, 300.0])
    for gene, vals in [("A", raw_A), ("B", raw_B), ("C", raw_C)]:
        a.X[:, list(a.var_names).index(gene)] = vals
    a.layers["imputed_count"] = a.X.copy()

    received_A = np.array([5.0, 6.0, 7.0])
    received_B = np.array([8.0, 9.0, 10.0])
    a.uns["received_ligands_tfl"] = pd.DataFrame(
        {"A": received_A, "B": received_B}, index=a.obs_names
    )

    metab_pairs = [("A", "B"), ("A", "C"), ("B", "B")]
    est = SpatialCellularProgramsEstimator(
        adata=a, target_gene=target, cluster_annot="cell_type_int",
        layer="imputed_count", radius=100, contact_distance=30,
        regulators=[], use_ligands=False, scale_factor=100,
        metab_pairs=metab_pairs,
    )

    assert est.metab_pairs == ["A@B", "A@C", "B@B"]

    est.init_data()
    frame = a.uns["metabolite_interactions"]
    assert list(frame.columns) == ["A@B", "A@C", "B@B"]

    np.testing.assert_array_equal(frame["A@B"].values, received_A * raw_B)
    np.testing.assert_array_equal(frame["A@C"].values, received_A * raw_C)
    np.testing.assert_array_equal(frame["B@B"].values, received_B * raw_B)


def test_metab_pairs_real_diffusion_path_populates_received_ligands_tfl():
    """No pre-seeded received_ligands_tfl this time: exercise the REAL
    `init_received_ligands` path (real CellChat DB load off the shipped CSV --
    no network required -- and the real cKDTree Gaussian kernel), not just the
    ad-hoc probe used by the other tests above. Confirms the metab export genes
    actually get diffused into `received_ligands_tfl` and the interaction
    frame is built correctly from that real output."""
    genes = ["A", "B"]
    target = "T"
    n = 8
    a = _make_adata(genes, target, n=n, seed=3)
    # deliberately NOT pre-seeding a.uns['received_ligands'] / ['received_ligands_tfl']

    metab_pairs = [("A", "B"), ("B", "A")]
    est = SpatialCellularProgramsEstimator(
        adata=a, target_gene=target, cluster_annot="cell_type_int",
        layer="imputed_count", radius=100, contact_distance=30,
        regulators=[], use_ligands=False, scale_factor=100,
        metab_pairs=metab_pairs,
    )
    est.init_data()

    assert "received_ligands_tfl" in a.uns
    tfl = a.uns["received_ligands_tfl"]
    assert "A" in tfl.columns and "B" in tfl.columns
    assert np.isfinite(tfl[["A", "B"]].values.astype(float)).all()

    frame = a.uns["metabolite_interactions"]
    assert list(frame.columns) == est.metab_pairs == ["A@B", "B@A"]

    counts = a.to_df(layer="imputed_count")
    np.testing.assert_allclose(frame["A@B"].values, tfl["A"].values * counts["B"].values)
    np.testing.assert_allclose(frame["B@A"].values, tfl["B"].values * counts["A"].values)


def test_metab_pairs_none_vs_omitted_no_op():
    """Passing metab_pairs=None explicitly is byte-identical to omitting the
    argument (proves the default preserves existing behavior). We compare
    both `modulators` and the resulting betadata *column contract*
    (`['beta0'] + ['beta_'+m for m in modulators]`, per get_betas' L1053)
    without training -- pure logic, no model fit."""
    genes = ["R1", "R2"]
    target = "T"
    a1 = _make_adata(genes, target, seed=2)
    a2 = _make_adata(genes, target, seed=2)

    est_none = SpatialCellularProgramsEstimator(
        adata=a1, target_gene=target, cluster_annot="cell_type_int",
        layer="imputed_count", radius=100, contact_distance=30,
        regulators=["R1", "R2"], use_ligands=False, scale_factor=100,
        metab_pairs=None,
    )
    est_omitted = SpatialCellularProgramsEstimator(
        adata=a2, target_gene=target, cluster_annot="cell_type_int",
        layer="imputed_count", radius=100, contact_distance=30,
        regulators=["R1", "R2"], use_ligands=False, scale_factor=100,
    )

    assert est_none.modulators == est_omitted.modulators == ["R1", "R2"]
    assert est_none.metab_pairs == est_omitted.metab_pairs == []
    assert est_none._diffusion_extra_lr == est_omitted._diffusion_extra_lr is None

    cols_none = ["beta0"] + ["beta_" + m for m in est_none.modulators]
    cols_omitted = ["beta0"] + ["beta_" + m for m in est_omitted.modulators]
    assert cols_none == cols_omitted == ["beta0", "beta_R1", "beta_R2"]


# ---------------------------------------------------------------------------
# Tier 1: real training (R^2 >= 0.15), exercises the group vector + fit path
# ---------------------------------------------------------------------------

def _build_metab_estimator(N=1200, seed=0):
    """Builds (but does not fit) the same estimator as
    `_trained_estimator_with_metab`, so tests can wrap `est.fit(...)` in a
    patch/spy (e.g. to capture the group-lasso `groups` array) before
    training."""
    rng = np.random.default_rng(seed)
    G = 14
    X = rng.random((N, G)).astype(np.float32)
    regs = [f"g{i}" for i in range(6)]
    target = "T"
    # g6, g7 double as the metabolite export/import genes (gA/gB below).
    names = regs + [f"g{i}" for i in range(6, G - 1)] + [target]
    X[:, -1] = (X[:, :6] @ rng.random(6)) + 0.02 * rng.standard_normal(N)

    gA, gB = "g6", "g7"

    a = ad.AnnData(X=X)
    a.var_names = names
    a.obs_names = [f"c{i}" for i in range(N)]
    a.obs["cell_type_int"] = pd.Categorical(rng.integers(0, 3, N))
    a.obsm["spatial"] = rng.uniform(0, 800, size=(N, 2))
    a.layers["imputed_count"] = X.copy()
    a.layers["normalized_count"] = X.copy()
    a.uns["received_ligands"] = pd.DataFrame(index=a.obs_names)
    # Seed received_ligands_tfl with the metab export genes (gA, gB) so
    # `received_ligands_tfl[self.metab_exports]` doesn't KeyError.
    a.uns["received_ligands_tfl"] = pd.DataFrame(
        {gA: X[:, names.index(gA)], gB: X[:, names.index(gB)]}, index=a.obs_names
    )

    links = {
        lb: pd.DataFrame({"source": regs, "target": [target] * 6,
                          "coef_mean": [0.3] * 6, "p": [1e-4] * 6})
        for lb in [0, 1, 2]
    }
    est = SpatialCellularProgramsEstimator(
        adata=a, target_gene=target, cluster_annot="cell_type_int",
        layer="imputed_count", radius=100, contact_distance=30,
        grn=RegulatoryFactory(links=links, annot="cell_type_int"),
        use_ligands=False, scale_factor=100,
        metab_pairs=[(gA, gB), (gB, gA)],
    )
    return est


def _trained_estimator_with_metab(N=1200, seed=0, num_epochs=25):
    """Mirrors tests/test_get_betas_batching.py::_trained_estimator, extended
    with a real `metab_pairs` group so training exercises group #5 too."""
    est = _build_metab_estimator(N=N, seed=seed)
    est.fit(num_epochs=num_epochs, use_pbar=False)
    return est


def test_metab_group_trains_and_emits_betas():
    est = _trained_estimator_with_metab()

    # a real CNN trained somewhere (not all zeroed-anchor fallbacks)
    assert any(v >= 0.15 for v in est.scores.values()), "fixture didn't train a real CNN"

    assert est.metab_pairs == ["g6@g7", "g7@g6"]

    bd = est.betadata
    assert "beta_g6@g7" in bd.columns
    assert "beta_g7@g6" in bd.columns
    assert np.isfinite(bd["beta_g6@g7"].values.astype(float)).all()
    assert np.isfinite(bd["beta_g7@g6"].values.astype(float)).all()

    # full betadata column contract preserved
    assert list(bd.columns) == ["beta0"] + ["beta_" + m for m in est.modulators]
    assert np.isfinite(bd.values.astype(float)).all()


def test_metab_group_uses_group_5_in_lasso():
    """Proves group #5 is what `fit()` actually hands to `GroupLasso` for the
    metab columns -- not just that `beta_<export>@<import>` columns exist
    afterward (which would still pass even if `fit()` mistakenly reused group
    4, since column naming/existence doesn't depend on the group-lasso group
    id at all). Patches `GroupLasso` (module-level import in
    parallel_estimators.py) to capture the `groups` kwarg it's constructed
    with, then reconstructs the expected vector from the estimator's own
    modulator-group lengths and checks the metab tail is exactly [5]*n."""
    est = _build_metab_estimator()

    captured = {}
    real_group_lasso = pe.GroupLasso

    class _CapturingGroupLasso(real_group_lasso):
        def __init__(self, *args, **kwargs):
            captured.setdefault("groups", kwargs.get("groups"))
            super().__init__(*args, **kwargs)

    with patch.object(pe, "GroupLasso", _CapturingGroupLasso):
        est.fit(num_epochs=1, use_pbar=False)

    groups = captured.get("groups")
    assert groups is not None, "GroupLasso was never constructed -- fixture didn't hit the lasso branch"

    expected = (
        [1] * len(est.regulators)
        + [2] * len(est.lr_pairs)
        + [3] * len(est.tfl_pairs)
        + [4] * len(est.extra_modulators)
        + [5] * len(est.metab_pairs)
    )
    assert len(groups) == len(est.modulators) == len(expected)
    assert list(groups) == expected
    # Specifically pin the metab tail to group 5: flipping `[5]` to `[4]` in
    # fit() would fail this even if the (weaker) full-vector check above were
    # accidentally satisfied some other way.
    assert len(est.metab_pairs) > 0
    assert list(groups[-len(est.metab_pairs):]) == [5] * len(est.metab_pairs)
