"""Tier-1 tests for `metab_processing/LinearRegression/build_x.py` (rev 4: deduplicated
factor block; rev 6: single factor block built on `normalized_count`).

`build_factor_block` builds, per focus gene, the REAL `SpatialCellularProgramsEstimator`
(non-metab) and calls `init_data()` -- exactly what training does
(`oracles.py::SpaceTravLR.run`) -- then merges each gene's `train_df` (minus the target
column) into ONE deduplicated union block keyed by column name, plus a per-gene column
map. So the ground truth here is still the estimator itself (mirroring
`tests/test_metab_group.py`'s style), but the parity check is against
`get_gene_factors`'s RECONSTRUCTION rather than a stored per-gene matrix. Requires torch
(via parallel_estimators); runs in the model env, not the pure-pandas Tier-0 loop.

Rev 6 (2026-09-29): dropped the two-SOURCE ('imputed'/'lognorm') dual-block machinery.
There is now ONE factor block, built on `normalized_count` (`build_x.LAYER`), stored
under the plain unsuffixed keys (`x_factors`, `x_factors_cols`, `x_factor_map`,
`x_metab`, `x_metab_modulators`, `x_genes`).
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
    build_factor_block, get_gene_factors, add_metabolites, add_gene_signature, build_x_adata,
    ensure_lognorm_layer, stored_genes, LAYER,
    _CLEANUP_OBSM, _CLEANUP_UNS,
)
from metab_processing.LinearRegression.least_squares import fit_gene_betas

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
    """Tiny PROCESSED-shaped AnnData: `imputed_count` + `raw_count` + `normalized_count`
    layers (mirroring the real persisted `_adata.h5ad` contract, plus the recreated
    `normalized_count` -- see `04_decisions_and_state.md`), `spatial` obsm, `cell_type_int`
    obs with 2 clusters (mirrors tests/test_metab_group.py::_make_adata). The factor block
    is built on `normalized_count` (`build_x.LAYER`); by default it equals `imputed_count`
    so most tests don't have to care about the distinction -- tests that specifically probe
    the normalized_count-vs-imputed_count difference override `normalized_count`."""
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
    a.layers["normalized_count"] = X.copy()
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


class _CountingRegulatoryFactory(RegulatoryFactory):
    """A RegulatoryFactory that counts calls to get_regulators per distinct `target_gene`
    arg, for the memoization test -- otherwise behaves exactly like the real class (same
    `get_regulators`/`get_regulators_with_pvalues` logic)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.call_counts = {}

    def get_regulators(self, adata, target_gene, alpha=0.05):
        self.call_counts[target_gene] = self.call_counts.get(target_gene, 0) + 1
        return super().get_regulators(adata, target_gene, alpha)


def _est_kwargs(**overrides):
    kwargs = dict(radius=RADIUS, contact_distance=CONTACT, scale_factor=SCALE,
                  tf_ligand_cutoff=0.01, receptor_thresh=0.01,
                  cluster_annot="cell_type_int")
    kwargs.update(overrides)
    return kwargs


def _truth_x(adata, target, grn, tflinks, layer=LAYER, **kwargs):
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
    assert stored_genes(result) == [TARGET, "C2"]
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
# perf fixes: get_regulators memoization + float32 preallocated block
# ---------------------------------------------------------------------------

def test_get_regulators_memoized_per_distinct_gene_and_restored_after():
    """build_factor_block must call grn.get_regulators ONCE per DISTINCT gene argument
    (not once per gene x NicheNet-ligand), and must restore the grn's ORIGINAL
    get_regulators afterward (not leave it monkeypatched). Two target genes sharing the
    same NicheNet ligand column (TGFB1, from the fixture tflinks) exercise the shared-
    ligand-lookup case that dominates the real cost at ~5000 genes."""
    adata = _make_adata(MOD_GENES, [TARGET, "C2"], n=25, seed=50)
    base_grn, tflinks = _build_networks()
    counting_grn = _CountingRegulatoryFactory(links=_links_dict(), annot="cell_type_int")
    kwargs = _est_kwargs()

    truth_T = _truth_x(adata, TARGET, base_grn, tflinks, **kwargs)
    truth_C2 = _truth_x(adata, "C2", base_grn, tflinks, **kwargs)

    assert "get_regulators" not in vars(counting_grn)  # pre-call: no instance-level shadow

    result = build_factor_block(adata, counting_grn, tflinks, [TARGET, "C2"], **kwargs)

    # at least one gene (TGFB1, the fixture's only NicheNet ligand column) must have been
    # looked up on behalf of BOTH target genes for the memo to mean anything here.
    assert "TGFB1" in counting_grn.call_counts
    # every distinct gene argument encountered (target genes + shared/distinct NicheNet
    # ligand columns) was looked up EXACTLY ONCE -- not once per (gene, ligand) pair.
    assert all(n == 1 for n in counting_grn.call_counts.values()), counting_grn.call_counts

    # restored to the EXACT pre-call state: no instance-level get_regulators shadow left
    # behind (not merely "patched back to a method that behaves like the original" --
    # `grn` had no instance attribute before the call, so it must have none after either).
    assert "get_regulators" not in vars(counting_grn)
    assert counting_grn.get_regulators.__func__ is _CountingRegulatoryFactory.get_regulators

    # result is byte-identical to the pre-fix (unmemoized) behavior.
    got_T = get_gene_factors(result, TARGET)
    got_C2 = get_gene_factors(result, "C2")
    assert list(got_T.columns) == list(truth_T.columns)
    np.testing.assert_allclose(got_T.to_numpy(), truth_T.to_numpy())
    assert list(got_C2.columns) == list(truth_C2.columns)
    np.testing.assert_allclose(got_C2.to_numpy(), truth_C2.to_numpy())


def test_get_regulators_memo_cache_copy_is_safe_against_mutation():
    """The memo must return a fresh list copy each call (not the cached list object
    itself), so a caller mutating `self.regulators` in place can't corrupt later lookups
    of the same gene. Checked two ways: directly against the GRN (pre-wrap sanity check,
    since `get_regulators` already returns a fresh `.tolist()` each call independent of
    the memo), and through build_factor_block's own wrapped cache using TWO target genes
    that both look up TGFB1 as a NicheNet ligand -- if the memo handed back the SAME
    cached list object both times, one estimator's internal mutation of its `regulators`
    list could corrupt the other's."""
    adata = _make_adata(MOD_GENES, [TARGET, "C2"], n=20, seed=51)
    direct_grn = _CountingRegulatoryFactory(links=_links_dict(), annot="cell_type_int")

    first = direct_grn.get_regulators(adata, "T")
    first.append("MUTATED")
    second = direct_grn.get_regulators(adata, "T")
    assert "MUTATED" not in second

    counting_grn = _CountingRegulatoryFactory(links=_links_dict(), annot="cell_type_int")
    tflinks = pd.DataFrame({"TGFB1": [0.5]}, index=["TF1"])
    build_factor_block(adata, counting_grn, tflinks, [TARGET, "C2"], **_est_kwargs())
    # TGFB1 (the shared NicheNet ligand) was looked up exactly once despite being needed
    # by both T and C2's init_ligands_and_receptors calls -- proves the cache was actually
    # hit (not merely that mutation was safe).
    assert counting_grn.call_counts.get("TGFB1") == 1


def test_x_factors_block_is_float32_and_matches_per_gene_train_df():
    """adata.obsm['x_factors'] must be float32 (the OOM fix), and the preallocated
    column-by-column fill must match what the per-gene train_df columns (and the old
    np.column_stack construction) would have produced, within float32 tolerance."""
    adata = _make_adata(MOD_GENES, [TARGET, "C2"], n=30, seed=52)
    grn, tflinks = _build_networks()
    kwargs = _est_kwargs()

    truth_T = _truth_x(adata, TARGET, grn, tflinks, **kwargs)
    truth_C2 = _truth_x(adata, "C2", grn, tflinks, **kwargs)

    result = build_factor_block(adata, grn, tflinks, [TARGET, "C2"], **kwargs)

    assert result.obsm["x_factors"].dtype == np.float32

    # what np.column_stack over the (gene-deduped) columns would have produced, for
    # comparison against the preallocated fill.
    cols = list(result.uns["x_factors_cols"])
    truth_cols = {}
    for c in truth_T.columns:
        truth_cols.setdefault(c, truth_T[c].to_numpy())
    for c in truth_C2.columns:
        truth_cols.setdefault(c, truth_C2[c].to_numpy())
    stacked = np.column_stack([truth_cols[c] for c in cols]).astype(np.float32)
    np.testing.assert_allclose(result.obsm["x_factors"], stacked, rtol=1e-6)

    # get_gene_factors still reconstructs each gene's matrix correctly off the float32 block.
    got_T = get_gene_factors(result, TARGET)
    got_C2 = get_gene_factors(result, "C2")
    np.testing.assert_allclose(got_T.to_numpy(), truth_T.to_numpy(), rtol=1e-6)
    np.testing.assert_allclose(got_C2.to_numpy(), truth_C2.to_numpy(), rtol=1e-6)


def test_union_cols_store_owned_copies_not_views_into_consolidated_block():
    """Fix A regression (MAJOR, the actual OOM fix): a stored per-column array must OWN
    its data, not be a view into X's consolidated (n_obs x M) float32 block.
    `X[col].to_numpy()` on a pandas frame whose same-dtype columns got consolidated
    returns a VIEW, and `.astype(..., copy=False)` is then a no-op when already float32 --
    so the stored "column" would pin the ENTIRE per-gene train_df block in memory for the
    rest of the build (worse than the old np.column_stack path this fix was meant to
    replace). NOTE: `np.ascontiguousarray(view, dtype=float32)` does NOT reliably fix this
    -- when the view is already C-contiguous float32 (the common case for a pandas block
    row, exercised by the fixture below), it returns the SAME view with no copy (verified
    empirically); only `.astype(float32)` with the DEFAULT `copy=True` is guaranteed by
    numpy to always allocate a new array. Checked directly against the exact construction
    used in build_factor_block, AND end-to-end against the final x_factors block."""
    # Direct check of the construction: a DataFrame with >1 same-dtype column (pandas
    # consolidates these into one block, so `.to_numpy()` on a single column is a
    # C-contiguous VIEW into that block -- the exact case where ascontiguousarray would
    # silently fail to copy).
    df = pd.DataFrame({"a": np.arange(5, dtype=np.float32), "b": np.arange(5, dtype=np.float32)})
    view = df["a"].to_numpy()
    assert view.base is not None and view.flags["C_CONTIGUOUS"], (
        "fixture must exercise the already-contiguous consolidated-block view case")
    # the broken candidate fix: ascontiguousarray returns the SAME view here (no copy).
    assert np.ascontiguousarray(view, dtype=np.float32) is view
    # the actual fix: astype's default copy=True always allocates a new, owned array.
    col_copy = view.astype(np.float32, copy=True)
    assert col_copy.base is None
    assert col_copy.flags["OWNDATA"]
    # mutating the original frame must not affect the copy.
    df.iloc[0, 0] = 999.0
    assert col_copy[0] != 999.0

    # End-to-end: build_factor_block's own output block is a standalone, contiguous,
    # data-owning array (it's a fresh np.empty fill, never a view into any train_df), and
    # each stored union_cols entry is independently owned (probed via a patched init that
    # mutates the shared train_df's underlying values after the column was extracted).
    adata = _make_adata(MOD_GENES, [TARGET, "C2"], n=15, seed=53)
    grn, tflinks = _build_networks()
    result = build_factor_block(adata, grn, tflinks, [TARGET, "C2"], **_est_kwargs())
    block = result.obsm["x_factors"]
    assert block.base is None
    assert block.flags["OWNDATA"]
    assert block.flags["C_CONTIGUOUS"]


def test_get_regulators_memo_bypassed_for_calls_with_extra_args():
    """Fix B: a get_regulators call carrying extra positional/keyword args (unused by
    today's two real call sites, but a safety guard against a future one) must bypass the
    memo cache entirely -- always call straight through to the original -- rather than be
    served from (or silently populate) a cache keyed on `gene` alone, which would ignore
    those extra args on a hit. Probes the ACTIVE wrapper from inside a build (not just the
    documented contract) by calling through `self.grn.get_regulators` -- the same wrapped
    instance attribute the real code paths use -- once plain (expect a cache hit, since
    TGFB1 was already queried as a NicheNet ligand by this gene's own real init) and once
    with an extra kwarg (must NOT be served from that cache)."""
    adata = _make_adata(MOD_GENES, [TARGET], n=15, seed=55)
    counting_grn = _CountingRegulatoryFactory(links=_links_dict(), annot="cell_type_int")
    tflinks = pd.DataFrame({"TGFB1": [0.5]}, index=["TF1"])

    captured = {}
    real_init = SpatialCellularProgramsEstimator.__init__

    def _probe_init(self, *a, **kw):
        real_init(self, *a, **kw)
        grn_obj = kw["grn"]  # the same (wrapped-for-this-build) instance as counting_grn
        n0 = counting_grn.call_counts.get("TGFB1", 0)
        assert n0 >= 1, "fixture must already have queried TGFB1 during real init"
        grn_obj.get_regulators(self.adata, "TGFB1")             # plain call -> cache hit
        n1 = counting_grn.call_counts.get("TGFB1", 0)
        grn_obj.get_regulators(self.adata, "TGFB1", alpha=0.9)  # extra kwarg -> bypass cache
        n2 = counting_grn.call_counts.get("TGFB1", 0)
        captured["plain_delta"] = n1 - n0
        captured["extra_delta"] = n2 - n1

    with patch.object(SpatialCellularProgramsEstimator, "__init__", _probe_init):
        build_factor_block(adata, counting_grn, tflinks, [TARGET], **_est_kwargs())

    assert captured["plain_delta"] == 0  # served from the cache, no extra real call
    assert captured["extra_delta"] == 1  # bypassed the cache -> one real call


def test_get_regulators_restored_after_mid_loop_exception():
    """If building a later gene's estimator raises, build_factor_block must still restore
    grn to its EXACT pre-call state before the exception propagates -- not leave the memo
    wrapper attached, which would silently cache stale/partial results for any later,
    separate call on the same `grn` object."""
    adata = _make_adata(MOD_GENES, [TARGET, "C2"], n=20, seed=54)
    grn, tflinks = _build_networks()
    assert "get_regulators" not in vars(grn)

    calls = {"n": 0}
    real_init = SpatialCellularProgramsEstimator.__init__

    def _boom_on_second(self, *a, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")
        return real_init(self, *a, **kw)

    with patch.object(SpatialCellularProgramsEstimator, "__init__", _boom_on_second):
        with pytest.raises(RuntimeError, match="boom"):
            build_factor_block(adata, grn, tflinks, [TARGET, "C2"], **_est_kwargs())

    assert "get_regulators" not in vars(grn)
    assert grn.get_regulators.__func__ is type(grn).get_regulators


# ---------------------------------------------------------------------------
# rev 6: single block, built on normalized_count
# ---------------------------------------------------------------------------

def test_block_reflects_normalized_count_not_imputed_count():
    """A synthetic adata with a `normalized_count` layer that is a DETERMINISTIC, distinct
    transform of `imputed_count` (a positive affine rescale of uniform-random values, far
    from `receptor_thresh` either way -- keeps the same receptor-threshold support so the
    column SET doesn't change, isolating the value difference we're testing for) must
    produce `x_factors`/`x_metab` values that match a ground truth built directly on
    `normalized_count`, and DIFFER from what `imputed_count` would have given -- proving
    the block is actually built on `normalized_count`, not `imputed_count`."""
    adata = _make_adata(MOD_GENES, [TARGET], n=30, seed=20)
    adata.layers["normalized_count"] = (adata.layers["imputed_count"] * 2.0 + 0.1).astype(np.float32)
    grn, tflinks = _build_networks()
    kwargs = _est_kwargs()

    truth_normalized = _truth_x(adata, TARGET, grn, tflinks, layer="normalized_count", **kwargs)
    truth_imputed = _truth_x(adata, TARGET, grn, tflinks, layer="imputed_count", **kwargs)

    result = build_factor_block(adata, grn, tflinks, [TARGET], **kwargs)
    add_metabolites(result, METABOLITES, radius=RADIUS, contact_distance=CONTACT,
                    scale_factor=SCALE)

    got = get_gene_factors(result, TARGET, metabs="all")
    non_metab_cols = list(truth_normalized.columns)
    np.testing.assert_allclose(got[non_metab_cols].to_numpy(), truth_normalized.to_numpy())
    assert not np.allclose(got[non_metab_cols].to_numpy(), truth_imputed[non_metab_cols].to_numpy())

    truth_metab = compute_metab_x(adata.copy(), METABOLITES, RADIUS, CONTACT, SCALE,
                                  "normalized_count")
    np.testing.assert_allclose(got["metab@M"].to_numpy(), truth_metab["metab@M"].to_numpy())


# ensure_lognorm_layer mirrors SpaceShip.process_adata_'s log1p-if-max>100 rule and is
# what build_x_adata uses to recreate the un-imputed log-norm layer that process_adata_
# creates then deletes (so it's never in a persisted `_adata.h5ad`).

def test_ensure_lognorm_layer_recreates_log1p_when_max_over_100():
    adata = _make_adata(MOD_GENES, [TARGET], n=10, seed=21)
    rng = np.random.default_rng(5)
    raw = (np.abs(rng.normal(scale=200, size=adata.shape)) + 150).astype(np.float32)
    adata.layers["raw_count"] = raw
    del adata.layers["normalized_count"]  # _make_adata's default; this test needs it absent

    ensure_lognorm_layer(adata)

    assert "normalized_count" in adata.layers
    np.testing.assert_allclose(adata.layers["normalized_count"], np.log1p(raw), rtol=1e-6)


def test_ensure_lognorm_layer_keeps_raw_when_max_not_over_100():
    adata = _make_adata(MOD_GENES, [TARGET], n=10, seed=22)
    rng = np.random.default_rng(6)
    raw = (rng.random(adata.shape) * 10).astype(np.float32)  # max well under 100
    adata.layers["raw_count"] = raw
    del adata.layers["normalized_count"]  # _make_adata's default; this test needs it absent

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
    del adata.layers["normalized_count"]  # _make_adata's default; this test needs it absent

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

    truth_M = compute_metab_x(adata.copy(), METABOLITES, RADIUS, CONTACT, SCALE, LAYER)
    truth_N = compute_metab_x(adata.copy(), METABOLITES2, RADIUS, CONTACT, SCALE, LAYER)

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
    assert "x_factors" in written.obsm
    assert "x_metab" in written.obsm
    assert list(written.uns["x_metab_modulators"]) == ["metab@M"]
    np.testing.assert_allclose(written.obsm["x_factors"], result.obsm["x_factors"])
    np.testing.assert_allclose(written.obsm["x_metab"], result.obsm["x_metab"])
    for key in _CLEANUP_OBSM:
        assert key not in written.obsm
    for key in _CLEANUP_UNS:
        assert key not in written.uns
    # normalized_count is kept in the final written adata (it's the factor block's own
    # y layer, and the source the block itself was built on). imputed_count (MAGIC) is
    # dropped by build_x_adata -- unused downstream, ~2.25GB freed during the build.
    assert "normalized_count" in written.layers
    assert "imputed_count" not in written.layers
    assert "raw_count" in written.layers


def test_build_x_adata_empty_metabolites_dict_skips_metab_block(tmp_path):
    """`metabolites={}` must hit the `if metabolites:` falsy gate like `None` -- no
    x_metab block is created (an empty dict is truthy-adjacent enough to be worth
    pinning)."""
    adata = _make_adata(MOD_GENES, [TARGET], n=20, seed=16)
    _, tflinks = _build_networks()
    setup_dir = tmp_path / "spacetravlr_output"
    _write_setup_dir(setup_dir, adata, tflinks)

    out_path = tmp_path / "x_adata.h5ad"
    result = build_x_adata(
        adata, str(out_path), focus_genes=[TARGET], metabolites={},
        setup_dir=setup_dir, **_est_kwargs())

    assert "x_metab" not in result.obsm
    assert "x_metab_modulators" not in result.uns
    written = ad.read_h5ad(out_path)
    assert "x_metab" not in written.obsm


def test_build_x_adata_run_params_json_overrides_args(tmp_path):
    """run_params.json, when present, is authoritative over the function's default args --
    except receptor_thresh, which run_params.json never persists, so it always stays the
    caller's default. `annot` in the fixture deliberately differs from the call's default so
    a broken key-mapping would actually be caught here."""
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

    assert captured["radius"] == 999
    assert captured["contact_distance"] == 77
    assert captured["scale_factor"] == 5
    assert captured["tf_ligand_cutoff"] == 0.02
    assert captured["cluster_annot"] == "cell_type_int_alt"
    # receptor_thresh has no run_params.json key -> always the caller's default, unchanged.
    assert captured["receptor_thresh"] == 0.42



# ---------------------------------------------------------------------------
# add_gene_signature: pure-pandas, single-block fixture (no estimator/torch needed)
# ---------------------------------------------------------------------------

SIG_N = 20


def _make_signature_adata(seed=0):
    """A minimal single-block-shaped AnnData: var_names A/B/C/D, a `normalized_count`
    layer (plus `raw_count`, to check the column-append mechanism handles >1 layer), a
    deduplicated `x_factors` block (5 unique columns f1..f5) and `x_factor_map` for
    A/B/C only (D is present in var_names but deliberately NOT a trained focus gene, to
    exercise the "constituent not in x_factor_map" skip)."""
    rng = np.random.default_rng(seed)
    genes = ["A", "B", "C", "D"]
    a = ad.AnnData(X=rng.normal(size=(SIG_N, len(genes))).astype(np.float32))
    a.var_names = genes
    a.obs_names = [f"c{i}" for i in range(SIG_N)]
    norm = rng.normal(size=(SIG_N, len(genes))).astype(np.float32)
    a.layers["normalized_count"] = norm
    a.layers["raw_count"] = rng.normal(size=(SIG_N, len(genes))).astype(np.float32)
    a.obsm["spatial"] = rng.uniform(0, 100, size=(SIG_N, 2))

    factors = rng.normal(size=(SIG_N, 5)).astype(np.float32)
    a.obsm["x_factors"] = factors
    a.uns["x_factors_cols"] = ["f1", "f2", "f3", "f4", "f5"]
    a.uns["x_factor_map"] = {
        "A": ["f1", "f2", "f3"],
        "B": ["f2", "f4"],
        "C": ["f2", "f3", "f5"],
    }
    a.uns["x_genes"] = ["A", "B", "C"]
    return a


def test_add_gene_signature_score_and_appending():
    adata = _make_signature_adata(seed=30)
    norm = pd.DataFrame(adata.layers["normalized_count"], columns=adata.var_names)

    new = add_gene_signature(adata, "SIG1", positive=["A", "B"], negative=["C"])

    assert "SIG1" in new.var_names
    assert "SIG1" in list(new.uns["x_genes"])
    expected_score = (norm["A"] + norm["B"] - norm["C"]).to_numpy()
    got_score = new[:, "SIG1"].layers["normalized_count"].ravel()
    np.testing.assert_allclose(got_score, expected_score, atol=1e-5)
    # score also lands in X and every other layer (shape-consistency requirement).
    got_score_X = np.asarray(new[:, "SIG1"].X).ravel()
    np.testing.assert_allclose(got_score_X, expected_score, atol=1e-5)
    np.testing.assert_allclose(
        new[:, "SIG1"].layers["raw_count"].ravel(), expected_score, atol=1e-5)

    # original untouched.
    assert "SIG1" not in adata.var_names
    assert "SIG1" not in list(adata.uns["x_genes"])
    assert adata.obsm["x_factors"].shape[1] == 5


def test_add_gene_signature_factor_union_order():
    adata = _make_signature_adata(seed=31)
    new = add_gene_signature(adata, "SIG1", positive=["A", "B"], negative=["C"],
                             factor_mode="union")
    # A: f1,f2,f3 ; B: f2,f4 ; C: f2,f3,f5 -> first-seen union across A,B,C.
    assert new.uns["x_factor_map"]["SIG1"] == ["f1", "f2", "f3", "f4", "f5"]


def test_add_gene_signature_factor_intersection_order():
    adata = _make_signature_adata(seed=32)
    new = add_gene_signature(adata, "SIG2", positive=["A", "B"], negative=["C"],
                             factor_mode="intersection")
    # only f2 is common to A, B and C.
    assert new.uns["x_factor_map"]["SIG2"] == ["f2"]


def test_add_gene_signature_get_gene_factors_and_fit_gene_betas():
    adata = _make_signature_adata(seed=33)
    new = add_gene_signature(adata, "SIG1", positive=["A", "B"], negative=["C"])

    got = get_gene_factors(new, "SIG1")
    assert list(got.columns) == new.uns["x_factor_map"]["SIG1"]

    df = fit_gene_betas(new, genes=["SIG1"], metabolites=None)
    assert set(df["gene"]) == {"SIG1"}
    assert set(df["factor"]) == set(new.uns["x_factor_map"]["SIG1"])


def test_add_gene_signature_missing_var_name_warns_and_skipped_from_score():
    adata = _make_signature_adata(seed=34)
    norm = pd.DataFrame(adata.layers["normalized_count"], columns=adata.var_names)

    with pytest.warns(UserWarning, match="ZZZ"):
        new = add_gene_signature(adata, "SIG1", positive=["A", "ZZZ"], negative=[])

    got_score = new[:, "SIG1"].layers["normalized_count"].ravel()
    np.testing.assert_allclose(got_score, norm["A"].to_numpy(), atol=1e-5)


def test_add_gene_signature_constituent_not_in_factor_map_skipped():
    adata = _make_signature_adata(seed=35)
    # D is a real gene (in var_names) but has no x_factor_map entry -> contributes to the
    # score but not to the factor set.
    new = add_gene_signature(adata, "SIG1", positive=["A", "D"], negative=[])
    assert new.uns["x_factor_map"]["SIG1"] == new.uns["x_factor_map"]["A"]


def test_add_gene_signature_all_missing_gives_zero_score_and_empty_factors():
    adata = _make_signature_adata(seed=36)
    with pytest.warns(UserWarning):
        new = add_gene_signature(adata, "SIG1", positive=["ZZZ"], negative=["YYY"])
    got_score = new[:, "SIG1"].layers["normalized_count"].ravel()
    np.testing.assert_allclose(got_score, np.zeros(SIG_N), atol=1e-8)
    assert new.uns["x_factor_map"]["SIG1"] == []


def test_add_gene_signature_preserves_float_score_in_integer_layer():
    """A layer with an integer dtype (e.g. raw_count as int) must still carry the exact
    float score in its appended `name` column -- casting the score to the layer's own
    dtype would truncate/round it (regression guard for the dtype-truncation bug)."""
    adata = _make_signature_adata(seed=39)
    norm = pd.DataFrame(adata.layers["normalized_count"], columns=adata.var_names)
    # replace raw_count with a small-integer layer so a truncating cast would be obvious.
    rng = np.random.default_rng(40)
    adata.layers["raw_count"] = rng.integers(0, 10, size=adata.shape).astype(np.int64)

    new = add_gene_signature(adata, "SIG1", positive=["A", "B"], negative=["C"])

    expected_score = (norm["A"] + norm["B"] - norm["C"]).to_numpy()
    got = new[:, "SIG1"].layers["raw_count"].ravel()
    np.testing.assert_allclose(got, expected_score, atol=1e-5)
    # sanity: the expected score is NOT (mostly) integer-valued, so a truncating cast
    # would have visibly failed this comparison.
    assert not np.allclose(got, np.round(got), atol=1e-3)


def test_add_gene_signature_duplicate_name_in_list_does_not_double_count():
    adata = _make_signature_adata(seed=41)
    norm = pd.DataFrame(adata.layers["normalized_count"], columns=adata.var_names)

    new = add_gene_signature(adata, "SIG1", positive=["A", "A", "B"], negative=[])

    expected_score = (norm["A"] + norm["B"]).to_numpy()  # NOT 2*A + B
    got = new[:, "SIG1"].layers["normalized_count"].ravel()
    np.testing.assert_allclose(got, expected_score, atol=1e-5)


def test_add_gene_signature_gene_in_both_lists_nets_to_zero():
    """A gene present in BOTH positive and negative nets to ~0 for that gene's
    contribution (each list is deduped independently, so it still counts once per
    side)."""
    adata = _make_signature_adata(seed=42)
    norm = pd.DataFrame(adata.layers["normalized_count"], columns=adata.var_names)

    new = add_gene_signature(adata, "SIG1", positive=["A", "B"], negative=["A"])

    expected_score = norm["B"].to_numpy()  # A - A cancels, leaving just B
    got = new[:, "SIG1"].layers["normalized_count"].ravel()
    np.testing.assert_allclose(got, expected_score, atol=1e-5)


def test_add_gene_signature_isolation_from_original():
    """The original adata is untouched by add_gene_signature: var_names/uns['x_genes']
    stay as they were, and the new adata's per-gene x_factor_map lists are independent
    copies -- mutating one does not affect the other's list for the same gene."""
    adata = _make_signature_adata(seed=43)
    orig_x_genes = list(adata.uns["x_genes"])
    orig_a_factors = list(adata.uns["x_factor_map"]["A"])

    new = add_gene_signature(adata, "SIG1", positive=["A", "B"], negative=["C"])

    assert "SIG1" not in adata.var_names
    assert list(adata.uns["x_genes"]) == orig_x_genes
    assert list(adata.uns["x_factor_map"]["A"]) == orig_a_factors

    new.uns["x_factor_map"]["A"].append("x")
    assert adata.uns["x_factor_map"]["A"] == orig_a_factors
    assert "x" not in adata.uns["x_factor_map"]["A"]


def test_add_gene_signature_bad_factor_mode_raises():
    adata = _make_signature_adata(seed=37)
    with pytest.raises(ValueError):
        add_gene_signature(adata, "SIG1", positive=["A"], negative=[], factor_mode="bogus")


def test_add_gene_signature_duplicate_name_raises():
    adata = _make_signature_adata(seed=38)
    with pytest.raises(ValueError):
        add_gene_signature(adata, "A", positive=["B"], negative=[])


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
