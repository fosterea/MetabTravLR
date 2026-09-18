"""Tier-1 tests for `metab_processing/LinearRegression/build_x.py` (rev 4: deduplicated
factor block; rev 5: two SOURCE blocks -- 'imputed' and 'lognorm' -- in one adata).

`build_factor_block` builds, per focus gene, the REAL `SpatialCellularProgramsEstimator`
(non-metab) and calls `init_data()` -- exactly what training does
(`oracles.py::SpaceTravLR.run`) -- then merges each gene's `train_df` (minus the target
column) into ONE deduplicated union block keyed by column name, plus a per-gene column
map. So the ground truth here is still the estimator itself (mirroring
`tests/test_metab_group.py`'s style), but the parity check is against
`get_gene_factors`'s RECONSTRUCTION rather than a stored per-gene matrix. Requires torch
(via parallel_estimators); runs in the model env, not the pure-pandas Tier-0 loop.

Rev 5: `source='imputed'|'lognorm'` selects which adata layer feeds the design matrix
(`SOURCE_LAYERS`).

Rev 5.1 (backward compat): `source='imputed'` reuses the ORIGINAL unsuffixed keys
(`x_factors`, `x_factors_cols`, `x_factor_map`, `x_metab`, `x_metab_modulators`,
`x_genes`) -- byte-identical to the pre-dual-block scheme, so an existing x_adata.h5ad
stays readable with no migration. Only `source='lognorm'` gets a `_lognorm` suffix on
every key (`x_factors_lognorm`, `x_factors_lognorm_cols`, `x_factor_map_lognorm`,
`x_metab_lognorm`, `x_metab_lognorm_modulators`, `x_genes_lognorm`). See
`build_x._SOURCE_SUFFIX`.
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
from metab_processing.SpaceTravLR.beta_analysis import _group, compute_metab_x
from metab_processing.LinearRegression import build_x as bx
from metab_processing.LinearRegression.build_x import (
    build_factor_block, get_gene_factors, add_metabolites, build_x_adata,
    ensure_lognorm_layer, stored_genes, SOURCE_LAYERS, _SOURCE_SUFFIX,
    _CLEANUP_OBSM, _CLEANUP_UNS,
)

RADIUS, CONTACT, SCALE = 100, 30, 100

# TGFB1/TGFBR1/TGFBR2 are real CellChat human L-R genes. A/B are a plain metabolite
# transporter pair. TF1 is a plain TF wired (via a hand-built grn) to regulate the target,
# and also wired (via a hand-built tflinks) as the top NicheNet regulator for TGFB1, so the
# L-TF group is non-trivial too. This lets one target gene exercise all four modulator
# groups (TF/L-R/L-TF via build_factor_block + metab via add_metabolites) at once.
MOD_GENES = ["TF1", "TGFB1", "TGFBR1", "TGFBR2", "A", "B"]
TARGET = "T"
METABOLITES = {"M": [("A", "B"), ("B", "A")]}
METABOLITES2 = {"N": [("A", "B")]}


def _make_adata(mod_genes, targets, n=30, seed=0):
    """Tiny PROCESSED-shaped AnnData: `imputed_count` + `raw_count` layers (mirroring the
    real persisted `_adata.h5ad` contract -- see `04_decisions_and_state.md`), `spatial`
    obsm, `cell_type_int` obs with 2 clusters (mirrors tests/test_metab_group.py::_make_adata).
    No `normalized_count` layer by default -- individual tests add one (or call
    `ensure_lognorm_layer`) when they need the 'lognorm' source."""
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
    a.layers["raw_count"] = X.copy()
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
                  tf_ligand_cutoff=0.01, receptor_thresh=0.01,
                  cluster_annot="cell_type_int")
    kwargs.update(overrides)
    return kwargs


def _truth_x(adata, target, grn, tflinks, layer="imputed_count", **kwargs):
    """Ground truth: an independently-built (metab-free) estimator's own train_df, on a
    fresh copy so it can't be affected by any uns mutation build_factor_block makes."""
    truth_adata = adata.copy()
    est = SpatialCellularProgramsEstimator(
        adata=truth_adata, target_gene=target, grn=grn, tflinks=tflinks,
        metabolites=None, layer=layer, **kwargs)
    est.init_data()
    return est.train_df.drop(columns=[target])


# ---------------------------------------------------------------------------
# build_factor_block: dedup + parity
# ---------------------------------------------------------------------------

def test_dedup_shared_lr_column_stored_once_and_reconstructs_exactly():
    """T and C2 both use the L-R group (target-gene-independent) -> shared columns must
    appear exactly once in the union block, and get_gene_factors must reconstruct each
    gene's matrix losslessly (values + column order) from that single copy."""
    adata = _make_adata(MOD_GENES, [TARGET, "C2"], n=30, seed=2)
    grn, tflinks = _build_networks()
    kwargs = _est_kwargs()

    truth_T = _truth_x(adata, TARGET, grn, tflinks, **kwargs)
    truth_C2 = _truth_x(adata, "C2", grn, tflinks, **kwargs)

    result = build_factor_block(adata, grn, tflinks, [TARGET, "C2"], **kwargs)

    assert result.uns["x_genes"] == [TARGET, "C2"]
    shared = [c for c in truth_T.columns if c in list(truth_C2.columns)]
    assert shared, "fixture must actually share >=1 column between T and C2"
    cols = list(result.uns["x_factors_cols"])
    for c in shared:
        assert cols.count(c) == 1, f"{c!r} must be stored exactly once (deduplicated)"

    got_T = get_gene_factors(result, TARGET)
    got_C2 = get_gene_factors(result, "C2")
    assert list(got_T.columns) == list(truth_T.columns)
    np.testing.assert_allclose(got_T.to_numpy(), truth_T.to_numpy())
    assert list(got_C2.columns) == list(truth_C2.columns)
    np.testing.assert_allclose(got_C2.to_numpy(), truth_C2.to_numpy())


def test_all_four_groups_present_via_get_gene_factors():
    """T has TF (TF1), L-R (TGFB1$TGFBR1/2), L-TF (TGFB1#TF1); add_metabolites supplies the
    4th. get_gene_factors(metabs='all') must expose all four (guards against a silently
    degraded fixture making this parity vacuous, like the rev-3 test)."""
    adata = _make_adata(MOD_GENES, [TARGET], n=40, seed=1)
    grn, tflinks = _build_networks()
    kwargs = _est_kwargs()

    result = build_factor_block(adata, grn, tflinks, [TARGET], **kwargs)
    add_metabolites(result, METABOLITES, radius=RADIUS, contact_distance=CONTACT,
                    scale_factor=SCALE)

    got = get_gene_factors(result, TARGET, metabs="all")
    groups = {_group(c) for c in got.columns}
    assert groups == {"tf", "lr", "ltf", "metab"}


# ---------------------------------------------------------------------------
# rev 5: two sources in one adata
# ---------------------------------------------------------------------------

def test_two_sources_produce_distinct_suffixed_blocks_with_different_values():
    """A synthetic adata with a `normalized_count` layer that is a DETERMINISTIC, distinct
    transform of `imputed_count` (a positive affine rescale of uniform-random values, far
    from `receptor_thresh` either way -- keeps the same receptor-threshold support so the
    two sources build the same column SET, isolating the value difference we're testing
    for) must produce different `x_factors` (the 'imputed' block -- unsuffixed, see
    backward-compat) vs `x_factors_lognorm` values for the same factor column, and
    different `x_metab` vs `x_metab_lognorm` values -- proving `source` actually drives
    which layer feeds the design matrix, not just a relabeling of the same numbers.
    (This calls build_factor_block directly with normalized_count already present, which
    is the ordering build_x_adata's BLOCKER fix specifically avoids for production use --
    see test_build_factor_block_imputed_gate_depends_on_normalized_count_presence and
    test_build_x_adata_imputed_block_gated_on_imputed_count below.)"""
    adata = _make_adata(MOD_GENES, [TARGET], n=30, seed=20)
    adata.layers["normalized_count"] = (adata.layers["imputed_count"] * 2.0 + 0.1).astype(np.float32)
    grn, tflinks = _build_networks()
    kwargs = _est_kwargs()

    result = build_factor_block(adata, grn, tflinks, [TARGET], source="imputed", **kwargs)
    build_factor_block(result, grn, tflinks, [TARGET], source="lognorm", **kwargs)

    # 'imputed' -> ORIGINAL unsuffixed keys (backward compat); 'lognorm' -> _lognorm.
    assert "x_factors" in result.obsm and "x_factors_lognorm" in result.obsm
    assert result.uns["x_factors_cols"] and result.uns["x_factors_lognorm_cols"]
    assert TARGET in result.uns["x_factor_map"]
    assert TARGET in result.uns["x_factor_map_lognorm"]

    got_imputed = get_gene_factors(result, TARGET, source="imputed")
    got_lognorm = get_gene_factors(result, TARGET, source="lognorm")
    assert list(got_imputed.columns) == list(got_lognorm.columns)
    assert not np.allclose(got_imputed.to_numpy(), got_lognorm.to_numpy())

    add_metabolites(result, METABOLITES, source="imputed", radius=RADIUS,
                    contact_distance=CONTACT, scale_factor=SCALE)
    add_metabolites(result, METABOLITES, source="lognorm", radius=RADIUS,
                    contact_distance=CONTACT, scale_factor=SCALE)
    assert "x_metab" in result.obsm and "x_metab_lognorm" in result.obsm
    assert list(result.uns["x_metab_modulators"]) == ["metab@M"]
    assert list(result.uns["x_metab_lognorm_modulators"]) == ["metab@M"]

    metab_imputed = get_gene_factors(result, TARGET, metabs="all", source="imputed")["metab@M"]
    metab_lognorm = get_gene_factors(result, TARGET, metabs="all", source="lognorm")["metab@M"]
    assert not np.allclose(metab_imputed.to_numpy(), metab_lognorm.to_numpy())

    # each source writes its OWN x_genes{sfx}; 'imputed' stays unsuffixed.
    assert result.uns["x_genes"] == [TARGET]
    assert result.uns["x_genes_lognorm"] == [TARGET]
    assert stored_genes(result, "imputed") == [TARGET]
    assert stored_genes(result, "lognorm") == [TARGET]


def test_get_gene_factors_source_lognorm_reads_lognorm_block():
    adata = _make_adata(MOD_GENES, [TARGET], n=25, seed=25)
    adata.layers["normalized_count"] = (adata.layers["imputed_count"] * 2.0 + 0.1).astype(np.float32)
    grn, tflinks = _build_networks()
    kwargs = _est_kwargs()

    build_factor_block(adata, grn, tflinks, [TARGET], source="lognorm", **kwargs)
    df = get_gene_factors(adata, TARGET, source="lognorm")
    assert list(df.columns) == adata.uns["x_factor_map_lognorm"][TARGET]
    # the 'imputed' block was never built -> asking for it must raise, not silently 404.
    with pytest.raises(KeyError, match="imputed"):
        get_gene_factors(adata, TARGET, source="imputed")


def test_get_gene_factors_unknown_source_raises_keyerror():
    adata = _make_adata(MOD_GENES, [TARGET], n=15, seed=26)
    grn, tflinks = _build_networks()
    build_factor_block(adata, grn, tflinks, [TARGET], **_est_kwargs())  # default source='imputed'

    with pytest.raises(KeyError):
        get_gene_factors(adata, TARGET, source="bogus")


def test_get_gene_factors_missing_source_block_raises_keyerror():
    """source='lognorm' requested but only 'imputed' was ever built -> a clear KeyError,
    not a silent empty result."""
    adata = _make_adata(MOD_GENES, [TARGET], n=15, seed=27)
    grn, tflinks = _build_networks()
    build_factor_block(adata, grn, tflinks, [TARGET], **_est_kwargs())  # 'imputed' only

    with pytest.raises(KeyError, match="lognorm"):
        get_gene_factors(adata, TARGET, source="lognorm")


def test_build_factor_block_unknown_source_raises_keyerror():
    adata = _make_adata(MOD_GENES, [TARGET], n=10, seed=28)
    grn, tflinks = _build_networks()
    with pytest.raises(KeyError):
        build_factor_block(adata, grn, tflinks, [TARGET], source="bogus", **_est_kwargs())


# ensure_lognorm_layer mirrors SpaceShip.process_adata_'s log1p-if-max>100 rule and is
# what build_x_adata uses to recreate the un-imputed log-norm layer that process_adata_
# creates then deletes (so it's never in a persisted `_adata.h5ad`).

def test_ensure_lognorm_layer_recreates_log1p_when_max_over_100():
    adata = _make_adata(MOD_GENES, [TARGET], n=10, seed=21)
    rng = np.random.default_rng(5)
    raw = (np.abs(rng.normal(scale=200, size=adata.shape)) + 150).astype(np.float32)
    adata.layers["raw_count"] = raw
    assert "normalized_count" not in adata.layers

    ensure_lognorm_layer(adata)

    assert "normalized_count" in adata.layers
    np.testing.assert_allclose(adata.layers["normalized_count"], np.log1p(raw), rtol=1e-6)


def test_ensure_lognorm_layer_keeps_raw_when_max_not_over_100():
    adata = _make_adata(MOD_GENES, [TARGET], n=10, seed=22)
    rng = np.random.default_rng(6)
    raw = (rng.random(adata.shape) * 10).astype(np.float32)  # max well under 100
    adata.layers["raw_count"] = raw

    ensure_lognorm_layer(adata)

    np.testing.assert_allclose(adata.layers["normalized_count"], raw)


def test_ensure_lognorm_layer_noop_if_already_present():
    adata = _make_adata(MOD_GENES, [TARGET], n=5, seed=23)
    existing = np.full(adata.shape, 7.0, dtype=np.float32)
    adata.layers["normalized_count"] = existing
    # a raw_count that WOULD give a different result if (wrongly) used instead.
    adata.layers["raw_count"] = np.full(adata.shape, 500.0, dtype=np.float32)

    ensure_lognorm_layer(adata)

    np.testing.assert_allclose(adata.layers["normalized_count"], existing)


def test_ensure_lognorm_layer_raises_without_raw_count():
    adata = _make_adata(MOD_GENES, [TARGET], n=5, seed=24)
    del adata.layers["raw_count"]
    assert "normalized_count" not in adata.layers

    with pytest.raises(ValueError, match="raw_count"):
        ensure_lognorm_layer(adata)


# ---------------------------------------------------------------------------
# build_factor_block: COMMOT refusal
# ---------------------------------------------------------------------------

def test_commot_cell_thresholds_raises_value_error():
    """COMMOT-masked diffusion (adata.uns['cell_thresholds']) is not supported: restoring
    received_ligands from received_ligands_tfl (the empty-frame guard) cannot reproduce a
    masked frame, so build_factor_block must refuse up front rather than silently compute
    wrong (unmasked) L-R values."""
    adata = _make_adata(MOD_GENES, [TARGET], n=20, seed=14)
    adata.uns["cell_thresholds"] = pd.DataFrame(index=adata.obs_names)
    grn, tflinks = _build_networks()

    with pytest.raises(ValueError, match="cell_thresholds"):
        build_factor_block(adata, grn, tflinks, [TARGET], **_est_kwargs())


# ---------------------------------------------------------------------------
# build_factor_block: empty-received_ligands guard (order independence)
# ---------------------------------------------------------------------------

def test_empty_received_ligands_guard_is_order_independent():
    """TGFB1 as an EARLY focus gene self-excludes its own (only) L-R pairs
    (parallel_estimators.py: `lr[~((lr.receptor==target)|(lr.ligand==target))]`), and since
    it is also the only NicheNet ligand column, self-exclusion empties its L-TF set too, and
    it has no TF regulators (grn only wires TF1->T) -- so TGFB1's own X has ZERO columns
    (skipped, warned) but its init_data() STILL sets `uns['received_ligands']` to an EMPTY
    frame (the real pre-existing bug at parallel_estimators.py ~991-997). Without the guard,
    the later gene T (real L-R) would KeyError reading `received_ligands[self.ligands]`."""
    adata = _make_adata(MOD_GENES, [TARGET], n=30, seed=3)
    grn, tflinks = _build_networks()
    kwargs = _est_kwargs()

    truth_T = _truth_x(adata, TARGET, grn, tflinks, **kwargs)
    assert any(c.split("$")[0] == "TGFB1" for c in truth_T.columns if "$" in c), (
        "fixture must actually give T real L-R columns for this guard to mean anything")

    with pytest.warns(UserWarning, match="no modulators"):
        result = build_factor_block(adata, grn, tflinks, ["TGFB1", TARGET], **kwargs)

    assert "TGFB1" not in result.uns["x_genes"]
    assert result.uns["x_genes"] == [TARGET]
    got_T = get_gene_factors(result, TARGET)
    assert list(got_T.columns) == list(truth_T.columns)
    np.testing.assert_allclose(got_T.to_numpy(), truth_T.to_numpy())


# ---------------------------------------------------------------------------
# get_gene_factors: metab selection
# ---------------------------------------------------------------------------

def test_get_gene_factors_metab_selection():
    adata = _make_adata(MOD_GENES, [TARGET], n=30, seed=4)
    grn, tflinks = _build_networks()
    kwargs = _est_kwargs()

    result = build_factor_block(adata, grn, tflinks, [TARGET], **kwargs)
    add_metabolites(result, METABOLITES, radius=RADIUS, contact_distance=CONTACT,
                    scale_factor=SCALE)
    base = get_gene_factors(result, TARGET)  # metabs=None baseline

    none_df = get_gene_factors(result, TARGET, metabs=None)
    assert list(none_df.columns) == list(base.columns)

    all_df = get_gene_factors(result, TARGET, metabs="all")
    assert "metab@M" in all_df.columns
    assert list(all_df.columns) == list(base.columns) + ["metab@M"]

    one_df = get_gene_factors(result, TARGET, metabs=["M"])
    assert list(one_df.columns) == list(base.columns) + ["metab@M"]
    np.testing.assert_allclose(
        one_df["metab@M"].to_numpy(), all_df["metab@M"].to_numpy())

    # a full "metab@M" spelling must also match
    prefixed_df = get_gene_factors(result, TARGET, metabs=["metab@M"])
    assert list(prefixed_df.columns) == list(one_df.columns)

    with pytest.warns(UserWarning, match="not found"):
        bad_df = get_gene_factors(result, TARGET, metabs=["ZZZ"])
    assert list(bad_df.columns) == list(base.columns)  # nothing appended

    empty_df = get_gene_factors(result, TARGET, metabs=[])
    assert list(empty_df.columns) == list(base.columns)  # empty list -> nothing appended


def test_get_gene_factors_missing_gene_raises_keyerror():
    adata = _make_adata(MOD_GENES, [TARGET], n=20, seed=8)
    grn, tflinks = _build_networks()
    result = build_factor_block(adata, grn, tflinks, [TARGET], **_est_kwargs())
    with pytest.raises(KeyError):
        get_gene_factors(result, "NOPE")


# ---------------------------------------------------------------------------
# add_metabolites: merge
# ---------------------------------------------------------------------------

def test_add_metabolites_merge_accumulates_and_dedupes():
    adata = _make_adata(MOD_GENES, [TARGET], n=25, seed=13)

    truth_M = compute_metab_x(adata.copy(), METABOLITES, RADIUS, CONTACT, SCALE, "imputed_count")
    truth_N = compute_metab_x(adata.copy(), METABOLITES2, RADIUS, CONTACT, SCALE, "imputed_count")

    add_metabolites(adata, METABOLITES, radius=RADIUS, contact_distance=CONTACT,
                    scale_factor=SCALE)
    assert list(adata.uns["x_metab_modulators"]) == ["metab@M"]

    add_metabolites(adata, METABOLITES2, radius=RADIUS, contact_distance=CONTACT,
                    scale_factor=SCALE)
    assert list(adata.uns["x_metab_modulators"]) == ["metab@M", "metab@N"]

    merged = pd.DataFrame(
        adata.obsm["x_metab"], columns=adata.uns["x_metab_modulators"],
        index=adata.obs_names)
    np.testing.assert_allclose(merged["metab@M"].to_numpy(), truth_M["metab@M"].to_numpy())
    np.testing.assert_allclose(merged["metab@N"].to_numpy(), truth_N["metab@N"].to_numpy())

    # re-adding an already-present metabolite name must NOT duplicate the column.
    add_metabolites(adata, METABOLITES, radius=RADIUS, contact_distance=CONTACT,
                    scale_factor=SCALE)
    assert list(adata.uns["x_metab_modulators"]) == ["metab@M", "metab@N"]
    assert adata.obsm["x_metab"].shape == (adata.n_obs, 2)


def test_add_metabolites_unknown_source_raises_keyerror():
    adata = _make_adata(MOD_GENES, [TARGET], n=10, seed=29)
    with pytest.raises(KeyError):
        add_metabolites(adata, METABOLITES, source="bogus", radius=RADIUS,
                        contact_distance=CONTACT, scale_factor=SCALE)


# ---------------------------------------------------------------------------
# build_factor_block: orphan / missing gene skip
# ---------------------------------------------------------------------------

def test_orphan_gene_skipped_and_warned():
    """A gene with no TF (grn only wires TF1 -> T, not ORPH), no L-R/L-TF (receptor_thresh
    set above every receptor's expression) -> 0-column X -> skipped + warned, not stored in
    uns['x_genes']."""
    adata = _make_adata(MOD_GENES, ["ORPH"], n=30, seed=5)
    grn, tflinks = _build_networks()

    with pytest.warns(UserWarning, match="no modulators"):
        result = build_factor_block(adata, grn, tflinks, ["ORPH"],
                                    **_est_kwargs(receptor_thresh=999))

    assert result.uns["x_genes"] == []
    assert result.uns["x_factors_cols"] == []
    assert result.obsm["x_factors"].shape == (adata.n_obs, 0)
    assert "ORPH" not in result.uns["x_factor_map"]


def test_orphan_only_x_factors_block_round_trips_through_h5ad(tmp_path):
    """An orphan-only panel leaves an (n_obs, 0) obsm['x_factors'] block; write_h5ad/
    read_h5ad must not drop or corrupt it, nor uns['x_factor_map']/x_factors_cols."""
    adata = _make_adata(MOD_GENES, ["ORPH"], n=15, seed=15)
    grn, tflinks = _build_networks()

    with pytest.warns(UserWarning, match="no modulators"):
        result = build_factor_block(adata, grn, tflinks, ["ORPH"],
                                    **_est_kwargs(receptor_thresh=999))

    out_path = tmp_path / "orphan.h5ad"
    result.write_h5ad(out_path)
    written = ad.read_h5ad(out_path)

    assert written.obsm["x_factors"].shape == (adata.n_obs, 0)
    assert list(written.uns["x_factors_cols"]) == []
    assert dict(written.uns["x_factor_map"]) == {}
    assert list(written.uns["x_genes"]) == []


def test_missing_gene_skipped_and_warned():
    adata = _make_adata(MOD_GENES, [TARGET], n=20, seed=9)
    grn, tflinks = _build_networks()

    with pytest.warns(UserWarning, match="not in adata.var_names"):
        result = build_factor_block(adata, grn, tflinks, [TARGET, "NOPE"], **_est_kwargs())

    assert "NOPE" not in result.uns["x_genes"]
    assert TARGET in result.uns["x_genes"]
    assert "NOPE" not in result.uns["x_factor_map"]


# ---------------------------------------------------------------------------
# build_factor_block: cleanup
# ---------------------------------------------------------------------------

def test_cleanup_pop_list_removes_diffusion_and_spatial_artifacts():
    """Mirrors build_x_adata's final cleanup: confirms the artifacts it pops actually exist
    after build_factor_block (guards a stale pop-list), and that popping them leaves the
    stored factor block intact."""
    adata = _make_adata(MOD_GENES, [TARGET], n=30, seed=6)
    grn, tflinks = _build_networks()
    build_factor_block(adata, grn, tflinks, [TARGET], **_est_kwargs())

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
    assert "x_factors" in adata.obsm  # the actual payload survives cleanup
    assert adata.uns["x_factor_map"]  # and the map survives too


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
    assert TARGET in written.uns["x_genes_lognorm"]
    for source, sfx in _SOURCE_SUFFIX.items():
        assert f"x_factors{sfx}" in written.obsm
        assert f"x_metab{sfx}" in written.obsm
        assert list(written.uns[f"x_metab{sfx}_modulators"]) == ["metab@M"]
        np.testing.assert_allclose(
            written.obsm[f"x_factors{sfx}"], result.obsm[f"x_factors{sfx}"])
        np.testing.assert_allclose(
            written.obsm[f"x_metab{sfx}"], result.obsm[f"x_metab{sfx}"])
    for key in _CLEANUP_OBSM:
        assert key not in written.obsm
    for key in _CLEANUP_UNS:
        assert key not in written.uns
    # backward compat: 'imputed' keys are the ORIGINAL unsuffixed ones.
    assert "x_factors" in written.obsm
    assert "x_metab" in written.obsm
    # normalized_count is kept in the final written adata (the 'lognorm' fits' y layer).
    assert "normalized_count" in written.layers
    assert "imputed_count" in written.layers
    assert "raw_count" in written.layers


def test_build_x_adata_empty_metabolites_dict_skips_metab_block(tmp_path):
    """`metabolites={}` must hit the `if metabolites:` falsy gate like `None` -- no x_metab
    block (for either source) is created (an empty dict is truthy-adjacent enough to be
    worth pinning)."""
    adata = _make_adata(MOD_GENES, [TARGET], n=20, seed=16)
    _, tflinks = _build_networks()
    setup_dir = tmp_path / "spacetravlr_output"
    _write_setup_dir(setup_dir, adata, tflinks)

    out_path = tmp_path / "x_adata.h5ad"
    result = build_x_adata(
        adata, str(out_path), focus_genes=[TARGET], metabolites={},
        setup_dir=setup_dir, **_est_kwargs())

    for source, sfx in _SOURCE_SUFFIX.items():
        assert f"x_metab{sfx}" not in result.obsm
        assert f"x_metab{sfx}_modulators" not in result.uns
    written = ad.read_h5ad(out_path)
    for source, sfx in _SOURCE_SUFFIX.items():
        assert f"x_metab{sfx}" not in written.obsm


def test_build_factor_block_imputed_gate_depends_on_normalized_count_presence():
    """BLOCKER root-cause regression: `init_ligands_and_receptors`
    (`parallel_estimators.py`) gates receptor/L-R selection on
    `'normalized_count' if present else 'imputed_count'` -- INDEPENDENT of the `layer=`
    build_factor_block passes to the estimator (which only controls VALUES). So building
    source='imputed' while a `normalized_count` layer happens to exist can silently change
    which L-R columns enter the block, even though every VALUE in the block still comes
    from `imputed_count`. This is exactly why `build_x_adata` must build 'imputed' BEFORE
    creating `normalized_count` (see the next test)."""
    grn, tflinks = _build_networks()
    kwargs = _est_kwargs(receptor_thresh=0.03)
    r1, r2 = MOD_GENES.index("TGFBR1"), MOD_GENES.index("TGFBR2")

    # Case A: normalized_count ABSENT. The gate must fall back to imputed_count, where
    # TGFBR1/2 are LOW (below receptor_thresh) -> TGFB1$TGFBR1/2 excluded.
    adata_a = _make_adata(MOD_GENES, [TARGET], n=30, seed=31)
    adata_a.layers["imputed_count"][:, r1] = 0.001
    adata_a.layers["imputed_count"][:, r2] = 0.001
    assert "normalized_count" not in adata_a.layers
    result_a = build_factor_block(adata_a, grn, tflinks, [TARGET], source="imputed", **kwargs)
    cols_a = result_a.uns["x_factor_map"].get(TARGET, [])

    # Case B: SAME imputed_count, but normalized_count is now PRESENT with TGFBR1/2 HIGH
    # (unrelated to imputed_count) -- reproduces what a naive (buggy) caller that creates
    # normalized_count BEFORE the imputed pass would hand to the estimator.
    adata_b = adata_a.copy()
    norm = adata_b.layers["imputed_count"].copy()
    norm[:, r1] = 5.0
    norm[:, r2] = 5.0
    adata_b.layers["normalized_count"] = norm
    result_b = build_factor_block(adata_b, grn, tflinks, [TARGET], source="imputed", **kwargs)
    cols_b = result_b.uns["x_factor_map"].get(TARGET, [])

    assert not any(c.startswith("TGFB1$") for c in cols_a), (
        "gated on imputed_count (LOW receptors) -> TGFB1$TGFBR* must be excluded")
    assert any(c.startswith("TGFB1$") for c in cols_b), (
        "normalized_count PRESENT (HIGH receptors) wrongly leaks into the 'imputed' "
        "block's gate -- this divergence is exactly what build_x_adata's build order "
        "must avoid")


def test_build_x_adata_imputed_block_gated_on_imputed_count(tmp_path):
    """BLOCKER regression, full production path: build_x_adata must build the 'imputed'
    block while normalized_count is ABSENT (popped if present from a reused setup dir), so
    its modulator SET is gated on imputed_count even though the FINAL written adata
    carries normalized_count (needed for the 'lognorm' fits' y). TGFBR1/2 are LOW in
    imputed_count but boosted in raw_count (so the recreated normalized_count, used for
    BOTH the 'lognorm' block's values and its own self-consistent gate, is HIGH) -- proving
    the two blocks' gates genuinely differ by source, and that 'imputed' is unaffected by
    the later-created normalized_count."""
    adata = _make_adata(MOD_GENES, [TARGET], n=30, seed=32)
    r1, r2 = MOD_GENES.index("TGFBR1"), MOD_GENES.index("TGFBR2")
    adata.layers["imputed_count"][:, r1] = 0.001
    adata.layers["imputed_count"][:, r2] = 0.001
    adata.layers["raw_count"][:, r1] = 5.0
    adata.layers["raw_count"][:, r2] = 5.0

    grn, tflinks = _build_networks()
    setup_dir = tmp_path / "spacetravlr_output"
    _write_setup_dir(setup_dir, adata, tflinks)

    out_path = tmp_path / "x_adata.h5ad"
    result = build_x_adata(
        adata, str(out_path), focus_genes=[TARGET], metabolites=None,
        setup_dir=setup_dir, **_est_kwargs(receptor_thresh=0.03))

    imputed_cols = result.uns["x_factor_map"].get(TARGET, [])
    lognorm_cols = result.uns["x_factor_map_lognorm"].get(TARGET, [])

    assert not any(c.startswith("TGFB1$") for c in imputed_cols), (
        "'imputed' block must be gated on imputed_count (LOW receptors), not a "
        "normalized_count that must not exist yet at that point")
    assert any(c.startswith("TGFB1$") for c in lognorm_cols), (
        "'lognorm' block, built after normalized_count is (re)created from the boosted "
        "raw_count, correctly includes the now-HIGH receptor pair")
    assert "normalized_count" in result.layers  # kept for the 'lognorm' fits' y


def test_build_x_adata_idempotent_on_reuse_with_preexisting_normalized_count(tmp_path):
    """A previously-written x_adata carrying normalized_count (e.g. re-running
    build_x_adata against a setup dir a prior run already touched) must have it POPPED
    before the imputed pass -- not silently reused as the imputed gate's source. Write a
    setup adata that ALREADY has a normalized_count layer with boosted TGFBR1/2 (as if a
    prior run's artifact leaked in); the 'imputed' block must still gate on imputed_count
    (LOW receptors -> excluded), proving the pop is unconditional, not presence-guarded."""
    adata = _make_adata(MOD_GENES, [TARGET], n=25, seed=33)
    r1, r2 = MOD_GENES.index("TGFBR1"), MOD_GENES.index("TGFBR2")
    adata.layers["imputed_count"][:, r1] = 0.001
    adata.layers["imputed_count"][:, r2] = 0.001
    stale_norm = adata.layers["imputed_count"].copy()
    stale_norm[:, r1] = 5.0
    stale_norm[:, r2] = 5.0
    adata.layers["normalized_count"] = stale_norm  # pre-existing, boosted receptors

    grn, tflinks = _build_networks()
    setup_dir = tmp_path / "spacetravlr_output"
    _write_setup_dir(setup_dir, adata, tflinks)

    out_path = tmp_path / "x_adata.h5ad"
    result = build_x_adata(
        adata, str(out_path), focus_genes=[TARGET], metabolites=None,
        setup_dir=setup_dir, **_est_kwargs(receptor_thresh=0.03))

    imputed_cols = result.uns["x_factor_map"].get(TARGET, [])
    assert not any(c.startswith("TGFB1$") for c in imputed_cols), (
        "a pre-existing normalized_count on the reused setup adata must be popped before "
        "the imputed pass, not left in place to leak into its gate")


def test_build_x_adata_run_params_json_overrides_args(tmp_path):
    """run_params.json, when present, is authoritative over the function's default args --
    except receptor_thresh, which run_params.json never persists, so it always stays the
    caller's default. `annot` in the fixture deliberately differs from the call's default so
    a broken key-mapping would actually be caught here. There is no `layer` override any
    more -- `source` (not a run_params.json key) now drives which layer is used."""
    adata = _make_adata(MOD_GENES, [TARGET], n=20, seed=11)
    adata.obs["cell_type_int_alt"] = adata.obs["cell_type_int"]
    _, tflinks = _build_networks()
    setup_dir = tmp_path / "spacetravlr_output"
    _write_setup_dir(setup_dir, adata, tflinks)
    run_params = {
        "radius": 999, "contact_distance": 77, "scale_factor": 5,
        "tf_ligand_cutoff": 0.02, "annot": "cell_type_int_alt",
    }
    (setup_dir / "betadata" / "run_params.json").write_text(json.dumps(run_params))

    captured = {}
    real = bx.build_factor_block

    def _spy(*args, **kw):
        captured.update(kw)
        return real(*args, **kw)

    out_path = tmp_path / "x_adata.h5ad"
    with patch.object(bx, "build_factor_block", side_effect=_spy):
        build_x_adata(
            adata, str(out_path), focus_genes=[TARGET], metabolites=None,
            setup_dir=setup_dir, radius=100, contact_distance=30, scale_factor=100,
            tf_ligand_cutoff=0.01, receptor_thresh=0.42,
            cluster_annot="cell_type_int")

    # build_factor_block is called once per source; the run_params-derived kwargs are
    # source-independent, so whichever call's kwargs `captured` holds last still proves
    # the override.
    assert captured["radius"] == 999
    assert captured["contact_distance"] == 77
    assert captured["scale_factor"] == 5
    assert captured["tf_ligand_cutoff"] == 0.02
    assert captured["cluster_annot"] == "cell_type_int_alt"
    # receptor_thresh has no run_params.json key -> always the caller's default, unchanged.
    assert captured["receptor_thresh"] == 0.42
    assert captured["source"] in SOURCE_LAYERS


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
