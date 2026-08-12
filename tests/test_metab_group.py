"""
Tests for the metabolite modulator group in `SpatialCellularProgramsEstimator`.

A metabolite is supplied as ``{name: [(export, import), ...]}`` and becomes ONE
summed modulator column:

    metab@<name> = sum over its (export, import) pairs of
                   received_ligand(export, diffused) * import(local)

Both orientations (a, b) and (b, a) are just two pairs in a metabolite's list and
are summed together -- directionality is intentionally dropped. It is its own
group-lasso group (#5); the betadata column carries a ``metab@`` marker (so the
read-back classifier still keys off ``@`` for the metab group) and is distinct
from L-R (``$``) and L-TF (``#``).

Tier 0 tests use `regulators=[]` (bypassing the CellOracle GRN entirely) and
`use_ligands=False` (bypassing CellChat) to stay self-contained and fast --
metabolite processing does not depend on either. Tier 1 (real training) mirrors
`tests/test_get_betas_batching.py::_trained_estimator`.
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


def _build_est(genes, target, metabolites, regulators=None, **kwargs):
    a = _make_adata(genes, target)
    est = SpatialCellularProgramsEstimator(
        adata=a, target_gene=target, cluster_annot="cell_type_int",
        layer="imputed_count", radius=100, contact_distance=30,
        regulators=[] if regulators is None else regulators,
        use_ligands=False, scale_factor=100,
        metabolites=metabolites,
        **kwargs,
    )
    return a, est


# ---------------------------------------------------------------------------
# Tier 0: pure-logic / no training
# ---------------------------------------------------------------------------

def test_column_construction_and_ordering():
    """metabolites -> one 'metab@<name>' column each, appended AFTER regulators
    (and after the -- here empty -- L-R/L-TF/extra groups) in `modulators`,
    preserving the dict's iteration order."""
    genes = ["R1", "R2", "A", "B", "C"]
    target = "T"
    metabolites = {"M1": [("A", "B"), ("B", "A")], "M2": [("C", "C")]}
    _, est = _build_est(genes, target, metabolites, regulators=["R1", "R2"])

    assert est.metab_pairs == ["metab@M1", "metab@M2"]
    assert est.modulators == ["R1", "R2", "metab@M1", "metab@M2"]
    assert est.modulators[-2:] == est.metab_pairs
    # summed pair-lists retained per metabolite
    assert est.metabolites == {"M1": [("A", "B"), ("B", "A")], "M2": [("C", "C")]}


def test_target_exclusion_drops_pair_keeps_metabolite():
    """A pair touching target_gene is excluded from the metabolite's SUM (a gene
    can't predict itself) but the metabolite survives on its other pairs, and the
    excluded pair still appears in the diffusion list (target-agnostic by design,
    so the shared received_ligands_tfl cache stays complete across genes)."""
    genes = ["A", "B", "C"]
    target = "T"
    metabolites = {"M": [("A", "T"), ("B", "C")]}
    _, est = _build_est(genes, target, metabolites)

    assert est.metab_pairs == ["metab@M"]
    assert est.metabolites == {"M": [("B", "C")]}
    assert est.metab_exports == ["B"]
    assert est.metab_imports == ["C"]
    # both still present in the diffusion-only list
    assert ("A", "T") in est._diffusion_extra_lr
    assert ("B", "C") in est._diffusion_extra_lr


def test_metabolite_all_pairs_target_touching_is_dropped():
    """A metabolite whose every pair touches the target gene contributes no
    column at all (dropped), but its pairs remain in the diffusion list."""
    genes = ["A", "B"]
    target = "T"
    metabolites = {"M": [("A", "T"), ("T", "B")]}
    _, est = _build_est(genes, target, metabolites)

    assert est.metab_pairs == []
    assert est.metabolites == {}
    assert ("A", "T") in est._diffusion_extra_lr and ("T", "B") in est._diffusion_extra_lr


def test_missing_gene_pair_dropped_metabolite_may_survive(capsys):
    """A pair referencing a gene not in adata.var_names is dropped (with a
    printed count); the metabolite survives on its remaining valid pairs."""
    genes = ["A", "B"]
    target = "T"
    metabolites = {"M": [("A", "B"), ("A", "Z")]}  # Z not a gene
    _, est = _build_est(genes, target, metabolites)

    assert est.metab_pairs == ["metab@M"]
    assert est.metabolites == {"M": [("A", "B")]}
    out = capsys.readouterr().out
    assert "1" in out and "metabolite pairs" in out


def test_dedup_exact_duplicate_within_metabolite():
    genes = ["A", "B"]
    target = "T"
    metabolites = {"M": [("A", "B"), ("A", "B")]}
    _, est = _build_est(genes, target, metabolites)

    assert est.metabolites == {"M": [("A", "B")]}


def test_type_validation_non_dict_raises():
    genes = ["A", "B"]
    target = "T"
    with pytest.raises(ValueError):
        _build_est(genes, target, metabolites=[("A", "B")])


def test_type_validation_non_str_key_raises():
    genes = ["A", "B"]
    target = "T"
    with pytest.raises(ValueError):
        _build_est(genes, target, metabolites={0: [("A", "B")]})


def test_type_validation_bad_pair_raises():
    genes = ["A", "B", "C"]
    target = "T"
    with pytest.raises(ValueError):
        _build_est(genes, target, metabolites={"M": [("A", "B", "C")]})


def test_init_data_summed_both_orientations_known_answer():
    """After init_data(), adata.uns['metabolite_interactions'] has EXACTLY one
    column per metabolite (metab@<name>) whose value is the SUM over the
    metabolite's pairs of received_ligands_tfl[export] * counts[import] (RAW,
    not diffused, on the import side). We pre-seed received_ligands_tfl with
    values distinct from raw X so the assertion distinguishes 'used
    received/diffused export' from 'used raw export', and covers the summing of
    both orientations (A,B)+(B,A) into a single column."""
    genes = ["A", "B"]
    target = "T"
    n = 4
    a = _make_adata(genes, target, n=n, seed=1)

    raw_A = np.array([1.0, 2.0, 3.0, 4.0])
    raw_B = np.array([10.0, 20.0, 30.0, 41.0])
    a.X[:, list(a.var_names).index("A")] = raw_A
    a.X[:, list(a.var_names).index("B")] = raw_B
    a.layers["imputed_count"] = a.X.copy()

    received_A = np.array([100.0, 200.0, 300.0, 400.0])
    received_B = np.array([1000.0, 2000.0, 3000.0, 4001.0])
    a.uns["received_ligands_tfl"] = pd.DataFrame(
        {"A": received_A, "B": received_B}, index=a.obs_names
    )

    metabolites = {"M": [("A", "B"), ("B", "A")]}
    est = SpatialCellularProgramsEstimator(
        adata=a, target_gene=target, cluster_annot="cell_type_int",
        layer="imputed_count", radius=100, contact_distance=30,
        regulators=[], use_ligands=False, scale_factor=100,
        metabolites=metabolites,
    )

    est.init_data()  # must not raise (exercises the column-count consistency assert)

    frame = a.uns["metabolite_interactions"]
    assert list(frame.columns) == est.metab_pairs == ["metab@M"]

    expected = received_A * raw_B + received_B * raw_A  # (A,B) + (B,A), summed
    np.testing.assert_array_equal(frame["metab@M"].values, expected)

    # train_df column order: target, regulators, L-R, L-TF, extra, metab
    assert list(est.train_df.columns) == [target, "metab@M"]


def test_init_data_multiple_metabolites_and_shared_genes_known_answer():
    """Two metabolites, one multi-pair, sharing an export gene: each column is
    the correct per-metabolite sum, with values computed from the correct
    (export, import) pairing (guards the per-pair alignment inside the sum)."""
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

    metabolites = {"M1": [("A", "B")], "M2": [("A", "C"), ("B", "B")]}
    est = SpatialCellularProgramsEstimator(
        adata=a, target_gene=target, cluster_annot="cell_type_int",
        layer="imputed_count", radius=100, contact_distance=30,
        regulators=[], use_ligands=False, scale_factor=100,
        metabolites=metabolites,
    )

    assert est.metab_pairs == ["metab@M1", "metab@M2"]

    est.init_data()
    frame = a.uns["metabolite_interactions"]
    assert list(frame.columns) == ["metab@M1", "metab@M2"]

    np.testing.assert_array_equal(frame["metab@M1"].values, received_A * raw_B)
    np.testing.assert_array_equal(
        frame["metab@M2"].values, received_A * raw_C + received_B * raw_B
    )


def test_metabolites_real_diffusion_path_populates_received_ligands_tfl():
    """No pre-seeded received_ligands_tfl this time: exercise the REAL
    `init_received_ligands` path (real CellChat DB load off the shipped CSV --
    no network required -- and the real cKDTree Gaussian kernel). Confirms the
    metab export genes actually get diffused into `received_ligands_tfl` and the
    summed interaction column is built correctly from that real output."""
    genes = ["A", "B"]
    target = "T"
    n = 8
    a = _make_adata(genes, target, n=n, seed=3)
    # deliberately NOT pre-seeding received_ligands / received_ligands_tfl

    metabolites = {"M": [("A", "B"), ("B", "A")]}
    est = SpatialCellularProgramsEstimator(
        adata=a, target_gene=target, cluster_annot="cell_type_int",
        layer="imputed_count", radius=100, contact_distance=30,
        regulators=[], use_ligands=False, scale_factor=100,
        metabolites=metabolites,
    )
    est.init_data()

    assert "received_ligands_tfl" in a.uns
    tfl = a.uns["received_ligands_tfl"]
    assert "A" in tfl.columns and "B" in tfl.columns
    assert np.isfinite(tfl[["A", "B"]].values.astype(float)).all()

    frame = a.uns["metabolite_interactions"]
    assert list(frame.columns) == est.metab_pairs == ["metab@M"]

    counts = a.to_df(layer="imputed_count")
    expected = tfl["A"].values * counts["B"].values + tfl["B"].values * counts["A"].values
    np.testing.assert_allclose(frame["metab@M"].values, expected)


def test_metabolites_none_vs_omitted_no_op():
    """Passing metabolites=None explicitly is byte-identical to omitting the
    argument (proves the default preserves existing behavior). We compare both
    `modulators` and the resulting betadata *column contract*
    (`['beta0'] + ['beta_'+m for m in modulators]`) without training."""
    genes = ["R1", "R2"]
    target = "T"
    a1 = _make_adata(genes, target, seed=2)
    a2 = _make_adata(genes, target, seed=2)

    est_none = SpatialCellularProgramsEstimator(
        adata=a1, target_gene=target, cluster_annot="cell_type_int",
        layer="imputed_count", radius=100, contact_distance=30,
        regulators=["R1", "R2"], use_ligands=False, scale_factor=100,
        metabolites=None,
    )
    est_omitted = SpatialCellularProgramsEstimator(
        adata=a2, target_gene=target, cluster_annot="cell_type_int",
        layer="imputed_count", radius=100, contact_distance=30,
        regulators=["R1", "R2"], use_ligands=False, scale_factor=100,
    )

    assert est_none.modulators == est_omitted.modulators == ["R1", "R2"]
    assert est_none.metab_pairs == est_omitted.metab_pairs == []
    assert est_none.metabolites == est_omitted.metabolites == {}
    assert est_none._diffusion_extra_lr == est_omitted._diffusion_extra_lr is None

    cols_none = ["beta0"] + ["beta_" + m for m in est_none.modulators]
    cols_omitted = ["beta0"] + ["beta_" + m for m in est_omitted.modulators]
    assert cols_none == cols_omitted == ["beta0", "beta_R1", "beta_R2"]


# ---------------------------------------------------------------------------
# Tier 1: real training (R^2 >= 0.15), exercises the group vector + fit path
# ---------------------------------------------------------------------------

def _build_metab_estimator(N=1200, seed=0, metabolites=None):
    """Builds (but does not fit) an estimator with a real metabolite group, so
    tests can wrap `est.fit(...)` in a patch/spy (e.g. to capture the group-lasso
    `groups` array) before training."""
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
    # Seed received_ligands_tfl with the metab export genes so
    # `received_ligands_tfl[self.metab_exports]` doesn't KeyError.
    a.uns["received_ligands_tfl"] = pd.DataFrame(
        {gA: X[:, names.index(gA)], gB: X[:, names.index(gB)]}, index=a.obs_names
    )

    links = {
        lb: pd.DataFrame({"source": regs, "target": [target] * 6,
                          "coef_mean": [0.3] * 6, "p": [1e-4] * 6})
        for lb in [0, 1, 2]
    }
    if metabolites is None:
        metabolites = {"MET": [(gA, gB), (gB, gA)]}
    est = SpatialCellularProgramsEstimator(
        adata=a, target_gene=target, cluster_annot="cell_type_int",
        layer="imputed_count", radius=100, contact_distance=30,
        grn=RegulatoryFactory(links=links, annot="cell_type_int"),
        use_ligands=False, scale_factor=100,
        metabolites=metabolites,
    )
    return est


def test_metab_group_trains_and_emits_betas():
    est = _build_metab_estimator()
    est.fit(num_epochs=25, use_pbar=False)

    # a real CNN trained somewhere (not all zeroed-anchor fallbacks)
    assert any(v >= 0.15 for v in est.scores.values()), "fixture didn't train a real CNN"

    assert est.metab_pairs == ["metab@MET"]

    bd = est.betadata
    assert "beta_metab@MET" in bd.columns
    assert np.isfinite(bd["beta_metab@MET"].values.astype(float)).all()

    # full betadata column contract preserved
    assert list(bd.columns) == ["beta0"] + ["beta_" + m for m in est.modulators]
    assert np.isfinite(bd.values.astype(float)).all()


def test_metab_group_uses_group_5_in_lasso():
    """Proves group #5 (one entry per metabolite) is what `fit()` hands to
    `GroupLasso` for the metab columns. Patches `GroupLasso` to capture the
    `groups` kwarg, then reconstructs the expected vector from the estimator's
    own modulator-group lengths and checks the metab tail is exactly [5]*n."""
    # Two metabolites -> exactly two group-5 entries.
    est = _build_metab_estimator(metabolites={"M1": [("g6", "g7")], "M2": [("g7", "g6")]})
    assert est.metab_pairs == ["metab@M1", "metab@M2"]

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
    assert len(est.metab_pairs) == 2
    assert list(groups[-len(est.metab_pairs):]) == [5] * len(est.metab_pairs)
