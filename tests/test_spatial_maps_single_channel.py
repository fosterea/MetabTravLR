"""Tests for the single-channel `spatial_maps` memory optimization.

Background: `adata.obsm['spatial_maps']` used to be `(N, n_clusters, 64, 64)`,
built by `xyc2spatial_fast`. Every channel was a bit-identical copy of the same
per-cell distance map (the per-cluster `mask` was `np.ones(...)`, a no-op), and
every consumer (`RotatedTensorDataset.__getitem__`, `get_betas` in
`models/parallel_estimators.py`) only ever read a single channel back out. So
storing `n_clusters` copies wasted `n_clusters`x memory for nothing.

`xyc2spatial_fast` now returns `(N, 1, 64, 64)` and consumers read channel 0.
This must be BIT-FOR-BIT behavior preserving: channel 0 of the new tensor must
equal every channel of the old tensor, and betas computed downstream must be
identical (not just close) to what the old C-channel path produced.

`_xyc2spatial_fast_multichannel_reference` (in `models/spatial_map.py`) is the
untouched, original multi-channel implementation, kept only so these tests can
reconstruct the "old" behavior and diff against it.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np
import pandas as pd
import anndata as ad
from unittest import TestCase

from SpaceTravLR.models.spatial_map import (
    xyc2spatial_fast,
    _xyc2spatial_fast_multichannel_reference,
)
from SpaceTravLR.tools.network import RegulatoryFactory
from SpaceTravLR.models.parallel_estimators import SpatialCellularProgramsEstimator


def _random_xyc(n_samples, n_clusters, clustered=False, seed=0):
    rng = np.random.default_rng(seed)
    if clustered:
        # coordinates clumped around a handful of centers, rather than uniform
        n_blobs = max(2, n_clusters)
        centers = rng.uniform(0, 1000, size=(n_blobs, 2))
        blob_ids = rng.integers(0, n_blobs, n_samples)
        xy = centers[blob_ids] + rng.normal(0, 15, size=(n_samples, 2))
    else:
        xy = rng.uniform(0, 1000, size=(n_samples, 2))
    labels = rng.integers(0, n_clusters, n_samples).astype(float)
    return np.column_stack([xy, labels])


class SingleChannelBitIdenticalTest(TestCase):
    """`xyc2spatial_fast` (new, 1-channel) must be bit-identical to every
    channel of the old multi-channel reference builder, across a variety of
    small random inputs."""

    def test_single_channel_matches_every_old_channel_uniform_coords(self):
        cases = [
            dict(n_samples=50, n_clusters=2, m=8, n=8, seed=1),
            dict(n_samples=300, n_clusters=5, m=16, n=16, seed=2),
            dict(n_samples=1500, n_clusters=7, m=32, n=32, seed=3),
            dict(n_samples=120, n_clusters=3, m=6, n=9, seed=4),  # non-square grid
        ]

        for case in cases:
            with self.subTest(**case):
                xyc = _random_xyc(
                    case["n_samples"], case["n_clusters"], clustered=False, seed=case["seed"]
                )
                new = xyc2spatial_fast(xyc, case["m"], case["n"])
                old = _xyc2spatial_fast_multichannel_reference(xyc, case["m"], case["n"])

                self.assertEqual(new.shape, (case["n_samples"], 1, case["m"], case["n"]))
                self.assertEqual(old.shape[0], case["n_samples"])
                self.assertEqual(old.shape[2:], (case["m"], case["n"]))

                for k in range(old.shape[1]):
                    self.assertTrue(
                        np.array_equal(new[:, 0], old[:, k]),
                        f"channel {k} differs from new single channel (case={case})",
                    )

    def test_single_channel_matches_every_old_channel_clustered_coords(self):
        cases = [
            dict(n_samples=200, n_clusters=4, m=12, n=12, seed=11),
            dict(n_samples=800, n_clusters=6, m=20, n=20, seed=12),
        ]

        for case in cases:
            with self.subTest(**case):
                xyc = _random_xyc(
                    case["n_samples"], case["n_clusters"], clustered=True, seed=case["seed"]
                )
                new = xyc2spatial_fast(xyc, case["m"], case["n"])
                old = _xyc2spatial_fast_multichannel_reference(xyc, case["m"], case["n"])

                self.assertEqual(new.shape, (case["n_samples"], 1, case["m"], case["n"]))

                for k in range(old.shape[1]):
                    self.assertTrue(
                        np.array_equal(new[:, 0], old[:, k]),
                        f"channel {k} differs from new single channel (case={case})",
                    )

    def test_new_shape_is_single_channel(self):
        xyc = _random_xyc(100, 3, seed=99)
        new = xyc2spatial_fast(xyc, 10, 10)
        self.assertEqual(new.shape, (100, 1, 10, 10))
        self.assertEqual(new.shape[1], 1)


def _trained_estimator(N=1200, n_clusters=3, seed=0, num_epochs=20):
    """Small, self-contained trained `SpatialCellularProgramsEstimator`, following
    the pattern in `tests/test_get_betas_batching.py`: `use_ligands=False`, a mock
    `RegulatoryFactory` grn, pre-seeded (empty) `received_ligands`/`received_ligands_tfl`
    frames, and a target that is a linear combination of regulators (+ small noise)
    so R2 >= 0.15 and a real CNN gets trained (not the zeroed-anchor fallback)."""
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
    a.obs["cell_type_int"] = pd.Categorical(rng.integers(0, n_clusters, N))
    a.obsm["spatial"] = rng.uniform(0, 800, size=(N, 2))
    a.layers["imputed_count"] = X.copy()
    a.layers["normalized_count"] = X.copy()
    a.uns["received_ligands"] = pd.DataFrame(index=a.obs_names)
    a.uns["received_ligands_tfl"] = pd.DataFrame(index=a.obs_names)

    links = {
        lb: pd.DataFrame({"source": regs, "target": [target] * 6,
                          "coef_mean": [0.3] * 6, "p": [1e-4] * 6})
        for lb in range(n_clusters)
    }
    est = SpatialCellularProgramsEstimator(
        adata=a, target_gene=target, cluster_annot="cell_type_int",
        layer="imputed_count", radius=100, contact_distance=30,
        grn=RegulatoryFactory(links=links, annot="cell_type_int"),
        use_ligands=False, scale_factor=100,
    )
    est.fit(num_epochs=num_epochs, use_pbar=False)
    return est


class BetasUnchangedEndToEndTest(TestCase):
    """Prove betas are IDENTICAL between the new 1-channel `sp_maps` path and a
    manually-tiled C-channel `sp_maps` fed through the OLD channel-slicing logic.
    The model must see exactly the same channel data either way, so the betas
    must be bitwise identical (not just close)."""

    def test_memory_shape_sanity(self):
        """`init_data` (and hence `adata.obsm['spatial_maps']`) now stores a
        single channel -- the whole point of the optimization."""
        est = _trained_estimator(N=400, n_clusters=3, num_epochs=1)
        self.assertEqual(est.adata.obsm["spatial_maps"].shape[1], 1)

    def test_get_betas_identical_to_old_tiled_channel_path(self):
        import torch

        est = _trained_estimator(N=1200, n_clusters=3, num_epochs=20)
        # sanity: at least one cluster actually trained a real CNN
        assert any(v >= 0.15 for v in est.scores.values()), "fixture didn't train a real CNN"

        # New path: whatever get_betas() does today (single-channel sp_maps,
        # sliced 0:1).
        new_betas = est.get_betas(betas_batch_size=4096)

        # Old path: manually reconstruct a tiled C-channel `sp_maps` (as
        # `xyc2spatial_fast` used to produce, pre-refactor) and run the OLD
        # cluster-indexed slicing logic (`cluster_target:cluster_target+1`)
        # by hand, batching + eval() exactly like the current get_betas does
        # (batch-size/eval-mode are orthogonal to this optimization -- see
        # test_get_betas_batching.py for that guarantee).
        n_clusters_old = int(np.max(est.cluster_labels)) + 1
        tiled_sp_maps = np.repeat(est.sp_maps, n_clusters_old, axis=1)
        self.assertEqual(tiled_sp_maps.shape[1], n_clusters_old)

        index_tracker = []
        betas = []
        betas_batch_size = 4096
        with torch.no_grad():
            for cluster_target in np.unique(est.cluster_labels):
                mask = est.cluster_labels == cluster_target
                indices = est.cell_indices[mask]
                index_tracker.extend(indices)

                if cluster_target not in est.models:
                    b = np.zeros((len(indices), len(est.modulators) + 1))
                else:
                    model = est.models[cluster_target]
                    model.eval()

                    # OLD slicing: pick the (bit-identical) channel belonging
                    # to this cluster out of the tiled C-channel tensor.
                    cluster_sp_maps_np = tiled_sp_maps[mask][:, cluster_target:cluster_target + 1, :, :]
                    spf_np = est.spatial_features.values[mask]

                    n_cells = cluster_sp_maps_np.shape[0]
                    b_chunks = []
                    for start in range(0, n_cells, betas_batch_size):
                        end = start + betas_batch_size
                        cluster_sp_maps = torch.from_numpy(
                            cluster_sp_maps_np[start:end]).float().to(est.device)
                        spf = torch.from_numpy(spf_np[start:end]).float().to(est.device)
                        b_chunk = model.get_betas(cluster_sp_maps, spf).cpu().numpy()
                        b_chunks.append(b_chunk)
                    b = np.concatenate(b_chunks, axis=0) if b_chunks else np.zeros(
                        (0, len(est.modulators) + 1))

                betas.extend(b)

        old_betas = pd.DataFrame(
            betas, index=index_tracker,
            columns=["beta0"] + ["beta_" + m for m in est.modulators],
        ).reindex(est.adata.obs.index)

        # THE key proof: bitwise identical, not just close.
        np.testing.assert_array_equal(new_betas.values, old_betas.values)
        self.assertListEqual(list(new_betas.columns), list(old_betas.columns))
        self.assertListEqual(list(new_betas.index), list(old_betas.index))

    def test_xyc2spatial_fast_feeds_identical_channel_data_to_model(self):
        """More direct version of the same proof at the `xyc2spatial_fast`
        level: build sp_maps via the new (1-channel) function and via the old
        multichannel reference, slice out "the same" channel for a given
        cluster from each, and confirm the raw arrays -- not just downstream
        betas -- are bit-identical."""
        xyc = _random_xyc(1000, 4, clustered=True, seed=7)
        new_sp_maps = xyc2spatial_fast(xyc, 64, 64)
        old_sp_maps = _xyc2spatial_fast_multichannel_reference(xyc, 64, 64)

        self.assertEqual(new_sp_maps.shape, (1000, 1, 64, 64))
        self.assertEqual(old_sp_maps.shape, (1000, 4, 64, 64))

        for cluster in range(4):
            new_channel = new_sp_maps[:, 0:1, :, :]
            old_channel = old_sp_maps[:, cluster:cluster + 1, :, :]
            np.testing.assert_array_equal(new_channel, old_channel)


if __name__ == "__main__":
    import unittest
    unittest.main(verbosity=2)
