"""Tests for the batched, eval-mode `get_betas` (GPU-OOM fix).

`SpatialCellularProgramsEstimator.get_betas()` used to run the CNN forward on an
ENTIRE cluster's cells at once, which OOMs on large datasets (the 112k-cell
Xenium run crashed here). It now batches the forward (`betas_batch_size`) and runs
in eval mode (BatchNorm running stats) so the result is batch-independent -- and
consistent with `predict()`, which already calls `.eval()`.

The central result-invariance guarantee: because eval-mode BatchNorm uses fixed
running statistics, the per-cell betas are identical regardless of batch size.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np
import pandas as pd
import anndata as ad
import torch

from SpaceTravLR.tools.network import RegulatoryFactory
from SpaceTravLR.models.parallel_estimators import SpatialCellularProgramsEstimator


def _trained_estimator(N=1200, seed=0, num_epochs=25):
    """A fitted estimator whose clusters train a REAL CNN (target = linear combo of
    regulators + small noise -> R2 >= 0.15, so we exercise the trained-model path, not the
    zeroed-anchor fallback). use_ligands=False + pre-seeded received-ligand frames keep it
    self-contained (no CellChat gene lookups).

    Seeds the GLOBAL torch/numpy RNGs (not just the local `rng` used for the
    synthetic data below) so this fixture -- and any test built on it -- is
    order-independent within the suite. Model weight init (CellularNicheNetwork),
    DataLoader shuffling, and RotatedTensorDataset's rotation (`np.random.choice`)
    all draw from the GLOBAL RNGs, not the local `np.random.default_rng(seed)`
    below; without this, a preceding test that also trains a model shifts those
    global RNGs and can tip loose-bound assertions here (observed: another test
    file's training run, executed first, changed this fixture's trained betas
    enough to fail `test_eval_mode_matches_old_train_mode_within_tiny_tolerance`'s
    1e-2 bound).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    G = 12
    X = rng.random((N, G)).astype(np.float32)
    regs = [f"g{i}" for i in range(6)]
    target = "T"
    names = regs + [f"g{i}" for i in range(6, G - 1)] + [target]
    X[:, -1] = (X[:, :6] @ rng.random(6)) + 0.02 * rng.standard_normal(N)

    a = ad.AnnData(X=X)
    a.var_names = names
    a.obs_names = [f"c{i}" for i in range(N)]
    a.obs["cell_type_int"] = pd.Categorical(rng.integers(0, 3, N))
    a.obsm["spatial"] = rng.uniform(0, 800, size=(N, 2))
    a.layers["imputed_count"] = X.copy()
    a.layers["normalized_count"] = X.copy()
    a.uns["received_ligands"] = pd.DataFrame(index=a.obs_names)
    a.uns["received_ligands_tfl"] = pd.DataFrame(index=a.obs_names)

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
    )
    est.fit(num_epochs=num_epochs, use_pbar=False)
    return est


def test_get_betas_is_batch_invariant():
    """THE key result-invariance test: identical betas across batch sizes (incl. 1 and >N)."""
    est = _trained_estimator()
    # sanity: at least one cluster actually trained a CNN (not all zeroed fallbacks)
    assert any(v >= 0.15 for v in est.scores.values()), "fixture didn't train a real CNN"

    ref = est.get_betas(betas_batch_size=4096).values.astype(np.float64)
    for bs in [1, 7, 100, 999, 10 ** 9]:
        got = est.get_betas(betas_batch_size=bs).values.astype(np.float64)
        assert got.shape == ref.shape
        # eval-mode BatchNorm is batch-independent -> results are (near-)identical
        assert np.allclose(got, ref, rtol=0, atol=1e-6), f"batch_size={bs} changed betas"


def test_get_betas_contract_unchanged():
    """Shape / columns / index of the returned DataFrame match the pre-change contract."""
    est = _trained_estimator()
    bd = est.get_betas()
    assert list(bd.index) == list(est.adata.obs.index)          # one row per cell, reindexed
    assert bd.shape[0] == est.adata.n_obs
    assert bd.columns[0] == "beta0"
    assert list(bd.columns) == ["beta0"] + ["beta_" + m for m in est.modulators]
    assert np.isfinite(bd.values.astype(float)).all()


def test_untrained_cluster_yields_zeros_block():
    """A cluster with no trained model still contributes a zeros block of the right width."""
    est = _trained_estimator()
    # drop one cluster's model to simulate the untrained/skip path
    dropped = sorted(est.models.keys())[0]
    del est.models[dropped]
    bd = est.get_betas()
    rows = est.adata.obs.index[np.asarray(est.cluster_labels) == dropped]
    block = bd.loc[rows].values.astype(float)
    assert block.shape[1] == len(est.modulators) + 1
    assert np.allclose(block, 0.0)


def test_eval_mode_matches_old_train_mode_within_tiny_tolerance():
    """Document the intended (tiny) behavior change: new eval-mode betas vs the OLD
    train-mode/whole-cluster forward. The diff shrinks as training converges (BatchNorm
    running stats approach the batch stats): measured max abs diff
        25 epochs -> ~3e-3,  60 -> ~3e-5,  150 (the run_spacetravlr default) -> ~3.6e-7.
    This fixture trains few epochs (fast), so we assert a loose bound; at production depth
    the difference is negligible (~1e-7)."""
    est = _trained_estimator()  # ~25 epochs
    import torch

    new_eval = est.get_betas().values.astype(np.float64)  # eval mode (current behavior)

    # reconstruct the OLD behavior: train-mode BatchNorm, whole cluster in one forward
    index_tracker, betas = [], []
    with torch.no_grad():
        for c in np.unique(est.cluster_labels):
            mask = est.cluster_labels == c
            index_tracker.extend(est.cell_indices[mask])
            if c not in est.models:
                betas.extend(np.zeros((int(mask.sum()), len(est.modulators) + 1)))
                continue
            m = est.models[c]
            m.train()  # old behavior: NOT eval
            # `sp_maps` now stores a single channel that is bit-identical to
            # the old per-cluster channel `c` (see `xyc2spatial_fast` in
            # `models/spatial_map.py`), so channel 0 here is what channel `c`
            # used to be.
            sm = torch.from_numpy(est.sp_maps[mask][:, 0:1, :, :]).float().to(est.device)
            sf = torch.from_numpy(est.spatial_features.values[mask]).float().to(est.device)
            betas.extend(m.get_betas(sm, sf).cpu().numpy())
    old_train = pd.DataFrame(
        betas, index=index_tracker,
        columns=["beta0"] + ["beta_" + i for i in est.modulators],
    ).reindex(est.adata.obs.index).values.astype(np.float64)

    # loose bound at this fixture's low epoch count; converges to ~1e-7 at production depth
    assert np.nanmax(np.abs(new_eval - old_train)) < 1e-2
