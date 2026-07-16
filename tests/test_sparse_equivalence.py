"""Correctness tests for the O(N^2) -> sparse/KDTree swap in
`SpaceTravLR.models.parallel_estimators`, and for the CU-5 kernel-width bugfix.

Two operations were made sparse:

1. `create_spatial_features` (per-cell-type neighbor COUNT within a radius): replaced the
   dense `cdist(coords, coords) <= radius` reference with a `cKDTree` radius query. This is
   a pure counting operation on a boolean `<= radius` mask, so the sparse version must match
   the dense reference **exactly** (`np.array_equal`), not just approximately. The dense
   reference implementation is kept as `_create_spatial_features_dense` for this purpose.
   UNCHANGED by this revision.

2. `received_ligands` / `compute_radius_weights_fast` (Gaussian-kernel-weighted ligand
   sum): replaced the dense N x N Gaussian weight matrix with a row-chunked `cKDTree`
   `sparse_distance_matrix` radius query.

   IMPORTANT: a prior revision of this sparse path (`compute_radius_weights_fast_dense`,
   now renamed `compute_radius_weights_fast_dense_wide_deprecated`) used the WRONG kernel:
   `sigma == radius` (no narrowing) and a truncation cutoff of `radius*sqrt(2*ln(1/eps))`
   (~6.4*radius at eps=1e-9). That is ~3.7x too wide vs. the original/intended kernel
   (`tools/utils.py::gaussian_kernel_2d`: `sigma = radius/sqrt(-2*ln(0.001))` i.e.
   `radius/3.7169`, hard cutoff at `dist <= radius`), and it is what caused the
   received-ligand sparse matrix to blow up in memory at Xenium density (radius=300 pulling
   in ~35k neighbors/cell, ~57 GB at 100k cells).

   The fix makes the sparse/fast path match `gaussian_kernel_2d` exactly. THIS DELIBERATELY
   CHANGES NUMERICAL RESULTS vs. the old wide kernel -- that is intended, not a regression.
   The correctness reference for these tests is therefore an explicit dense computation
   built directly from `gaussian_kernel_2d` (`_dense_narrow_reference` below), NOT the old
   `compute_radius_weights_fast_dense_wide_deprecated` (which is kept only for historical
   comparison and is asserted nowhere in this file).

Per `03_dev_process_and_testing.md` ("over-test on purpose"): this file sweeps many random
configurations rather than a handful of hand-picked ones, because these two ops are exercised
once per training run and any silent divergence would corrupt every downstream beta.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree

from SpaceTravLR.models.parallel_estimators import (
    compute_radius_weights_fast,
    compute_radius_weights_fast_sparse,
    create_spatial_features,
    received_ligands,
    _create_spatial_features_dense,
    _sparse_gaussian_kernel_2d,
)
from SpaceTravLR.tools.utils import gaussian_kernel_2d

RNG_SEED = 12345


def _clustered_coords(n, n_clusters=4, spread=5.0, extent=1000.0, seed=0):
    """A handful of tight Gaussian blobs, as a stand-in for real tissue spatial structure
    (as opposed to the uniform-random coordinates used elsewhere)."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0, extent, size=(n_clusters, 2))
    assignments = rng.integers(0, n_clusters, size=n)
    return centers[assignments] + rng.normal(0, spread, size=(n, 2))


def _dense_narrow_reference(xy, lig_df, radius, scale_factor):
    """THE correctness reference for the received-ligand kernel: builds the narrow,
    hard-cutoff-at-`radius` Gaussian kernel row-by-row using `gaussian_kernel_2d` directly
    (the original/intended semantics), then applies the exact same normalization
    (`scale_factor / N`) as `compute_radius_weights_fast_sparse`. O(N^2) -- fine for test
    sizes, not meant for production."""
    xy = np.ascontiguousarray(xy, dtype=np.float64)
    lig_values = np.ascontiguousarray(lig_df.values, dtype=np.float64)
    n = len(lig_df)
    result = np.zeros((n, lig_values.shape[1]), dtype=np.float64)
    for i in range(n):
        w = gaussian_kernel_2d(xy[i], xy, radius=radius)  # eps=0.001 default, dist<=radius
        result[i] = (scale_factor / n) * (w @ lig_values)
    return pd.DataFrame(result, index=lig_df.index, columns=lig_df.columns)


# ---------------------------------------------------------------------------
# 1. create_spatial_features: EXACT equivalence (counting op, no tolerance needed)
#    -- UNCHANGED by this revision, kept as a regression guard.
# ---------------------------------------------------------------------------

class TestSpatialFeaturesExactEquivalence:
    def test_exact_equivalence_across_many_configurations(self):
        rng = np.random.default_rng(RNG_SEED)
        n_configs = 0
        for N in [50, 500, 2000]:
            for n_ct in [1, 2, 4, 8]:
                for radius in [1, 25, 200, 5000]:
                    for coord_kind in ["uniform_float", "clustered_float", "uniform_int"]:
                        if coord_kind == "uniform_float":
                            x = rng.uniform(0, 1000, N)
                            y = rng.uniform(0, 1000, N)
                        elif coord_kind == "clustered_float":
                            coords = _clustered_coords(
                                N, n_clusters=min(6, max(2, n_ct)),
                                seed=int(rng.integers(0, 1_000_000)),
                            )
                            x, y = coords[:, 0], coords[:, 1]
                        else:  # uniform_int -- integer coordinates
                            x = rng.integers(0, 1000, N).astype(float)
                            y = rng.integers(0, 1000, N).astype(float)

                        celltypes = rng.integers(0, n_ct, N)
                        obs_index = pd.Index([f"cell_{i}" for i in range(N)])

                        dense = _create_spatial_features_dense(
                            x, y, celltypes, obs_index, radius=radius
                        )
                        sparse = create_spatial_features(
                            x, y, celltypes, obs_index, radius=radius
                        )

                        assert list(dense.columns) == list(sparse.columns)
                        assert dense.index.equals(sparse.index)
                        assert dense.shape == sparse.shape
                        assert np.array_equal(dense.values, sparse.values), (
                            f"MISMATCH N={N} n_ct={n_ct} radius={radius} "
                            f"coords={coord_kind}"
                        )
                        n_configs += 1
        assert n_configs >= 100

    def test_exact_equivalence_at_exact_boundary_distance(self):
        # Points placed at EXACTLY `radius` apart -- this is where an off-by-epsilon or a
        # strict-vs-non-strict inequality bug would show up (dense uses `<=`).
        radius = 5.0
        x = np.array([0.0, 3.0, 0.0, 10.0])
        y = np.array([0.0, 4.0, -4.0, 10.0])  # dist(0,1)=5.0, dist(0,2)=4.0 exactly
        celltypes = np.array([0, 0, 1, 1])
        obs_index = pd.Index(["a", "b", "c", "d"])
        dense = _create_spatial_features_dense(x, y, celltypes, obs_index, radius=radius)
        sparse = create_spatial_features(x, y, celltypes, obs_index, radius=radius)
        assert np.array_equal(dense.values, sparse.values)


# ---------------------------------------------------------------------------
# 2. received-ligand kernel: tight-tolerance equivalence vs. the NARROW reference
# ---------------------------------------------------------------------------

class TestReceivedLigandKernelEquivalence:
    def test_sparse_vs_narrow_dense_reference_across_many_configurations(self):
        rng = np.random.default_rng(RNG_SEED + 1)
        max_rel_err = 0.0
        n_configs = 0
        for N in [30, 150, 600, 2000]:
            for n_lig in [1, 3, 6]:
                for radius in [5, 50, 300]:
                    for scale_factor in [1, 100]:
                        for coord_kind in ["uniform", "clustered"]:
                            if coord_kind == "uniform":
                                xy = rng.uniform(0, 1000, size=(N, 2))
                            else:
                                xy = _clustered_coords(
                                    N, n_clusters=4, spread=8.0,
                                    seed=int(rng.integers(0, 1_000_000)),
                                )

                            lig = pd.DataFrame(
                                rng.random((N, n_lig)),
                                index=[f"cell_{i}" for i in range(N)],
                                columns=[f"Lig{i}" for i in range(n_lig)],
                            )

                            ref = _dense_narrow_reference(xy, lig, radius, scale_factor)
                            sparse = compute_radius_weights_fast_sparse(
                                xy, lig, radius, scale_factor
                            )

                            assert ref.index.equals(sparse.index)
                            assert list(ref.columns) == list(sparse.columns)
                            assert ref.shape == sparse.shape

                            rel = np.abs(ref.values - sparse.values) / np.maximum(
                                np.abs(ref.values), 1e-12
                            )
                            max_rel_err = max(max_rel_err, float(rel.max()))
                            assert np.allclose(
                                ref.values, sparse.values, rtol=1e-6, atol=1e-8
                            ), f"MISMATCH N={N} n_lig={n_lig} radius={radius} sf={scale_factor} coords={coord_kind}"
                            n_configs += 1
        assert n_configs >= 50
        # Sanity print (visible with `pytest -s`) -- worst-case relative error over the sweep.
        print(f"\n[test_sparse_equivalence] max relative error vs narrow reference over sweep = {max_rel_err:.3e}")

    def test_equivalence_holds_at_large_magnitude(self):
        """Guards that equivalence is governed by RELATIVE tolerance, not the absolute
        atol used in the sweep. Absolute error scales with scale_factor*ligand, so at
        large magnitude (values ~1e9 here) absolute diffs can be non-trivial, yet the sparse
        result still matches the narrow reference to rtol=1e-6."""
        rng = np.random.default_rng(RNG_SEED + 7)
        xy = _clustered_coords(400, n_clusters=5, spread=10.0, seed=123)
        lig = pd.DataFrame(
            rng.uniform(500, 1500, size=(400, 4)),
            index=[f"cell_{i}" for i in range(400)],
            columns=[f"Lig{i}" for i in range(4)],
        )
        radius, scale_factor = 100.0, 1e6
        ref = _dense_narrow_reference(xy, lig, radius, scale_factor)
        sparse = compute_radius_weights_fast_sparse(xy, lig, radius, scale_factor)
        # atol=0.0 -> pure relative comparison; proves rtol is what actually protects us
        assert np.allclose(ref.values, sparse.values, rtol=1e-6, atol=0.0)
        rel = np.abs(ref.values - sparse.values) / np.maximum(np.abs(ref.values), 1e-12)
        assert rel.max() < 1e-6


# ---------------------------------------------------------------------------
# 3. Kernel-semantics pin test -- so the width/cutoff can't silently drift again
# ---------------------------------------------------------------------------

class TestKernelSemanticsPin:
    def test_weight_at_dist_zero_is_one(self):
        radius = 10.0
        xy = np.array([[0.0, 0.0], [50.0, 50.0]])
        W = _sparse_gaussian_kernel_2d(xy, radius, query_xy=xy[:1])
        row = np.asarray(W.todense()).flatten()
        assert np.isclose(row[0], 1.0)

    def test_weight_at_dist_equal_to_radius_is_approx_eps(self):
        # gaussian_kernel_2d: sigma = radius/sqrt(-2*ln(eps)) so that, by construction,
        # weight(dist == radius) == eps (default eps = 0.001).
        radius = 25.0
        xy = np.array([[0.0, 0.0], [radius, 0.0]])
        W = _sparse_gaussian_kernel_2d(xy, radius, eps=0.001, query_xy=xy[:1])
        row = np.asarray(W.todense()).flatten()
        assert np.isclose(row[1], 0.001, rtol=1e-3)

    def test_weight_just_beyond_radius_is_excluded(self):
        radius = 25.0
        xy = np.array([[0.0, 0.0], [radius + 1e-3, 0.0]])
        W = _sparse_gaussian_kernel_2d(xy, radius, query_xy=xy[:1])
        row = np.asarray(W.todense()).flatten()
        assert row[1] == 0.0

    def test_weight_matches_gaussian_kernel_2d_directly(self):
        """Cross-check against `gaussian_kernel_2d` itself at a sweep of distances,
        including exactly at the boundary."""
        rng = np.random.default_rng(99)
        radius = 40.0
        origin = np.array([0.0, 0.0])
        for dist in [0.0, 1.0, 10.0, radius - 1e-6, radius, radius + 1e-6, 100.0]:
            xy = np.array([origin, [dist, 0.0]])
            expected = gaussian_kernel_2d(origin, xy, radius=radius)[1]
            W = _sparse_gaussian_kernel_2d(xy, radius, query_xy=xy[:1])
            got = np.asarray(W.todense()).flatten()[1]
            assert np.isclose(got, expected, rtol=1e-6, atol=1e-12), (
                f"dist={dist}: expected {expected}, got {got}"
            )

    def test_isolated_cell_received_ligand_equals_scale_over_n_times_self(self):
        """A single isolated cell (only itself within the cutoff) must receive exactly
        scale_factor/N * lig[self] -- no neighbor contribution leaks in."""
        N = 8
        radius = 1.0
        rng = np.random.default_rng(5)
        # widely spaced along a line -- pairwise distances all >> radius
        xy = np.array([[i * 1e6, 0.0] for i in range(N)])
        lig_values = rng.uniform(1, 10, N)
        lig = pd.DataFrame({"L1": lig_values}, index=[f"c{i}" for i in range(N)])
        sf = 7.0
        sparse = compute_radius_weights_fast_sparse(xy, lig, radius, sf)
        expected = sf / N * lig_values
        assert np.allclose(sparse["L1"].values, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# 4. Chunking invariance
# ---------------------------------------------------------------------------

class TestChunkingInvariance:
    @pytest.mark.parametrize("chunk_size", [1, 3, 7, 50, 2000, 100_000])
    def test_chunked_equals_unchunked(self, chunk_size):
        rng = np.random.default_rng(RNG_SEED + 11)
        N = 250
        xy = rng.uniform(0, 500, size=(N, 2))
        lig = pd.DataFrame(
            rng.random((N, 3)),
            index=[f"cell_{i}" for i in range(N)],
            columns=["A", "B", "C"],
        )
        radius, sf = 60.0, 12.0
        unchunked = compute_radius_weights_fast_sparse(xy, lig, radius, sf, chunk_size=N)
        chunked = compute_radius_weights_fast_sparse(xy, lig, radius, sf, chunk_size=chunk_size)
        # each row's result depends only on that row's own neighbor query, independent of
        # chunk boundaries -- so this must be EXACT, not merely close.
        assert np.array_equal(unchunked.values, chunked.values)

    def test_chunked_equals_unchunked_clustered_coords(self):
        xy = _clustered_coords(300, n_clusters=5, spread=6.0, seed=321)
        lig = pd.DataFrame(
            np.random.default_rng(2).random((300, 2)),
            index=[f"c{i}" for i in range(300)],
            columns=["L1", "L2"],
        )
        radius, sf = 40.0, 3.0
        full = compute_radius_weights_fast_sparse(xy, lig, radius, sf, chunk_size=300)
        for cs in [1, 17, 64, 299, 301]:
            chunked = compute_radius_weights_fast_sparse(xy, lig, radius, sf, chunk_size=cs)
            assert np.array_equal(full.values, chunked.values), f"chunk_size={cs} diverged"

    def test_compute_radius_weights_fast_default_chunk_size_matches_full(self):
        """The public entry point (`compute_radius_weights_fast`, used by `received_ligands`)
        defaults to a chunk_size (8000) smaller than typical test N's -- confirm that still
        matches an explicit single-chunk call."""
        rng = np.random.default_rng(RNG_SEED + 12)
        N = 500
        xy = rng.uniform(0, 500, size=(N, 2))
        lig = pd.DataFrame(rng.random((N, 2)), index=[f"c{i}" for i in range(N)], columns=["A", "B"])
        default_call = compute_radius_weights_fast(xy, lig, 30.0, 5.0)
        single_chunk = compute_radius_weights_fast_sparse(xy, lig, 30.0, 5.0, chunk_size=N)
        assert np.array_equal(default_call.values, single_chunk.values)


# ---------------------------------------------------------------------------
# 5. Memory sanity -- guards against re-widening the kernel/cutoff
# ---------------------------------------------------------------------------

class TestMemorySanity:
    def test_xenium_density_neighbor_count_is_bounded(self):
        """At Xenium-like density (median nearest-neighbor spacing ~15) with radius=300 (a
        typical secreted-signaling radius), the number of neighbors within the cutoff must
        be O(1e3), not O(N) -- this is exactly the blowup the fix addresses. Uses a regular
        grid at spacing 15 as a stand-in for Xenium-like cell packing."""
        N = 5000
        spacing = 15.0
        side = int(np.ceil(np.sqrt(N)))
        xs, ys = np.meshgrid(np.arange(side) * spacing, np.arange(side) * spacing)
        xy = np.column_stack([xs.ravel(), ys.ravel()])[:N]

        radius = 300.0
        tree = cKDTree(xy)
        # count_neighbors counts ALL ordered pairs (i, j) with dist(i, j) <= radius,
        # including self-pairs (i == i) and both (i, j) and (j, i).
        total_pairs = tree.count_neighbors(tree, r=radius)
        avg_neighbors = total_pairs / N

        print(f"\n[memory sanity] N={N}, radius={radius}, avg neighbors/cell = {avg_neighbors:.1f}")

        # Guards against re-widening: with the OLD (wide) kernel this cutoff would have been
        # ~6.4x radius (~1920), pulling in the entire 5000-cell grid per cell. With the fixed
        # cutoff == radius, expected neighbors ~ pi*radius^2 / spacing^2 ~ 1257 -- far below
        # N/2.
        assert avg_neighbors < N / 2, (
            f"avg_neighbors={avg_neighbors:.1f} is not << N={N}; the cutoff may have widened again"
        )
        # Also a positive floor -- this should not be near-zero (i.e. the radius query isn't
        # accidentally returning almost nothing).
        assert avg_neighbors > 10


# ---------------------------------------------------------------------------
# 6. Edge cases
# ---------------------------------------------------------------------------

class TestReceivedLigandEdgeCases:
    def test_single_cell(self):
        xy = np.array([[10.0, 20.0]])
        lig = pd.DataFrame({"L1": [3.0]}, index=["c0"])
        radius, sf = 5.0, 2.0
        ref = _dense_narrow_reference(xy, lig, radius, sf)
        sparse = compute_radius_weights_fast_sparse(xy, lig, radius, sf)
        expected = sf / 1 * 3.0  # only self, weight 1, N=1
        assert np.isclose(ref.loc["c0", "L1"], expected)
        assert np.isclose(sparse.loc["c0", "L1"], expected, atol=1e-12)

    def test_all_cells_at_same_location_everyone_is_a_neighbor(self):
        N = 6
        xy = np.zeros((N, 2))
        lig_values = np.arange(1, N + 1, dtype=float)
        lig = pd.DataFrame({"L1": lig_values}, index=[f"c{i}" for i in range(N)])
        radius, sf = 10.0, 1.0
        ref = _dense_narrow_reference(xy, lig, radius, sf)
        sparse = compute_radius_weights_fast_sparse(xy, lig, radius, sf)
        # every pairwise distance is 0 -> every weight is 1 -> received[i] = sf/N * sum(lig)
        expected = sf / N * lig_values.sum()
        assert np.allclose(ref["L1"].values, expected)
        assert np.allclose(sparse["L1"].values, expected, atol=1e-10)

    def test_cells_all_far_apart_only_self_within_cutoff(self):
        N = 5
        radius = 1.0
        xy = np.array([[i * 1_000_000.0, 0.0] for i in range(N)])
        lig_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lig = pd.DataFrame({"L1": lig_values}, index=[f"c{i}" for i in range(N)])
        sf = 3.0
        sparse = compute_radius_weights_fast_sparse(xy, lig, radius, sf)
        # spec: received[i] = scale_factor/N * lig[i] when only self is within cutoff
        expected = sf / N * lig_values
        assert np.allclose(sparse["L1"].values, expected, atol=1e-15)

    def test_zero_ligand_values(self):
        N = 20
        rng = np.random.default_rng(3)
        xy = rng.uniform(0, 500, size=(N, 2))
        lig = pd.DataFrame(
            {"L1": np.zeros(N), "L2": np.zeros(N)}, index=[f"c{i}" for i in range(N)]
        )
        ref = _dense_narrow_reference(xy, lig, 50.0, 10.0)
        sparse = compute_radius_weights_fast_sparse(xy, lig, 50.0, 10.0)
        assert np.allclose(ref.values, 0.0)
        assert np.allclose(sparse.values, 0.0, atol=1e-12)

    def test_duplicate_coordinates(self):
        N = 10
        rng = np.random.default_rng(4)
        half = rng.uniform(0, 200, size=(N // 2, 2))
        xy = np.vstack([half, half])  # exact duplicate coordinates
        lig = pd.DataFrame({"L1": rng.random(N)}, index=[f"c{i}" for i in range(N)])
        ref = _dense_narrow_reference(xy, lig, 30.0, 5.0)
        sparse = compute_radius_weights_fast_sparse(xy, lig, 30.0, 5.0)
        assert np.allclose(ref.values, sparse.values, rtol=1e-6, atol=1e-8)

    def test_one_ligand_column(self):
        N = 12
        rng = np.random.default_rng(11)
        xy = rng.uniform(0, 300, size=(N, 2))
        lig = pd.DataFrame({"OnlyLig": rng.random(N)}, index=[f"c{i}" for i in range(N)])
        ref = _dense_narrow_reference(xy, lig, 40.0, 2.0)
        sparse = compute_radius_weights_fast_sparse(xy, lig, 40.0, 2.0)
        assert np.allclose(ref.values, sparse.values, rtol=1e-6, atol=1e-8)


class TestSpatialFeaturesEdgeCases:
    def test_single_cell(self):
        x, y = np.array([5.0]), np.array([5.0])
        celltypes = np.array([0])
        idx = pd.Index(["c0"])
        dense = _create_spatial_features_dense(x, y, celltypes, idx, radius=10)
        sparse = create_spatial_features(x, y, celltypes, idx, radius=10)
        assert np.array_equal(dense.values, sparse.values)
        assert dense.values[0, 0] == 1  # sees only itself

    def test_one_cell_type(self):
        N = 15
        rng = np.random.default_rng(5)
        x = rng.uniform(0, 100, N)
        y = rng.uniform(0, 100, N)
        celltypes = np.zeros(N, dtype=int)
        idx = pd.Index([f"c{i}" for i in range(N)])
        dense = _create_spatial_features_dense(x, y, celltypes, idx, radius=40)
        sparse = create_spatial_features(x, y, celltypes, idx, radius=40)
        assert np.array_equal(dense.values, sparse.values)
        assert dense.shape[1] == 1

    def test_a_celltype_absent_from_the_population(self):
        # np.unique(celltypes) never yields a label with zero members (verified: even a
        # pandas Categorical with an unused category collapses via np.unique), so a
        # "0_within"-style all-zero column can't appear here by construction. What we CAN
        # (and do) verify: leaving one label entirely out of the data doesn't crash and
        # produces exactly the columns that ARE present, identically in both paths.
        N = 12
        rng = np.random.default_rng(6)
        x = rng.uniform(0, 100, N)
        y = rng.uniform(0, 100, N)
        celltypes = rng.choice([0, 1], size=N)  # label "2" never appears
        idx = pd.Index([f"c{i}" for i in range(N)])
        dense = _create_spatial_features_dense(x, y, celltypes, idx, radius=30)
        sparse = create_spatial_features(x, y, celltypes, idx, radius=30)
        assert np.array_equal(dense.values, sparse.values)
        assert list(dense.columns) == ["0_within", "1_within"]

    def test_duplicate_coordinates(self):
        N = 10
        rng = np.random.default_rng(7)
        half = rng.uniform(0, 200, size=(N // 2, 2))
        xy = np.vstack([half, half])
        x, y = xy[:, 0], xy[:, 1]
        celltypes = rng.integers(0, 3, N)
        idx = pd.Index([f"c{i}" for i in range(N)])
        dense = _create_spatial_features_dense(x, y, celltypes, idx, radius=25)
        sparse = create_spatial_features(x, y, celltypes, idx, radius=25)
        assert np.array_equal(dense.values, sparse.values)


# ---------------------------------------------------------------------------
# 7. Determinism + the 1/N-not-1/k normalization guard
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_create_spatial_features_is_deterministic(self):
        rng = np.random.default_rng(8)
        N = 200
        x = rng.uniform(0, 500, N)
        y = rng.uniform(0, 500, N)
        celltypes = rng.integers(0, 4, N)
        idx = pd.Index([f"c{i}" for i in range(N)])
        r1 = create_spatial_features(x, y, celltypes, idx, radius=60)
        r2 = create_spatial_features(x, y, celltypes, idx, radius=60)
        assert np.array_equal(r1.values, r2.values)

    def test_received_ligand_kernel_is_deterministic(self):
        rng = np.random.default_rng(9)
        N = 150
        xy = rng.uniform(0, 500, size=(N, 2))
        lig = pd.DataFrame(
            rng.random((N, 4)), index=[f"c{i}" for i in range(N)], columns=list("ABCD")
        )
        r1 = compute_radius_weights_fast_sparse(xy, lig, 40.0, 10.0)
        r2 = compute_radius_weights_fast_sparse(xy, lig, 40.0, 10.0)
        assert np.array_equal(r1.values, r2.values)  # exact determinism, not just close


class TestNormalizationIsByTotalCellCountNotNeighborCount:
    """Targeted regression: the spec is explicit that `1/N` uses the FULL cell count, not
    the neighbor count. This test is built so that, if someone "helpfully" switched the
    normalization to divide by each cell's own neighbor count instead of N, the isolated
    cells below would come out N times too large -- easy to miss with only random inputs
    where N and neighbor-count sometimes coincide."""

    def test_isolated_cells_are_scaled_by_1_over_N_not_1_over_neighbor_count(self):
        # 3 cells tightly clustered (mutual neighbors within radius) + 2 cells isolated far
        # away (each sees ONLY itself within the cutoff). N = 5 for everyone.
        xy = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.1],
                [0.0, 0.2],
                [10_000.0, 10_000.0],
                [20_000.0, 20_000.0],
            ]
        )
        N = 5
        radius = 1.0  # cutoff == radius, way smaller than the inter-group spacing
        scale_factor = 2.0
        lig_values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        lig = pd.DataFrame({"L1": lig_values}, index=[f"c{i}" for i in range(N)])

        sparse = compute_radius_weights_fast_sparse(xy, lig, radius, scale_factor)
        ref = _dense_narrow_reference(xy, lig, radius, scale_factor)

        # Isolated cell c3: only itself is within the cutoff (neighbor count == 1), but the
        # correct divisor is the TOTAL cell count N == 5.
        correct_value = scale_factor / N * lig_values[3]
        wrong_value_if_divided_by_neighbor_count = scale_factor / 1 * lig_values[3]

        assert np.isclose(sparse.loc["c3", "L1"], correct_value, atol=1e-12)
        assert np.isclose(ref.loc["c3", "L1"], correct_value, atol=1e-8)
        assert not np.isclose(sparse.loc["c3", "L1"], wrong_value_if_divided_by_neighbor_count)

        # Same check for cell c4.
        correct_value_c4 = scale_factor / N * lig_values[4]
        assert np.isclose(sparse.loc["c4", "L1"], correct_value_c4, atol=1e-12)

        # Clustered cell c0: neighbor count == 3 (itself + c1 + c2), but divisor must still
        # be N == 5, not 3.
        # weight(c0->c1) uses gaussian_kernel_2d's sigma = radius/sqrt(-2*ln(0.001))
        sigma = radius / np.sqrt(-2.0 * np.log(0.001))
        w01 = np.exp(-(0.1 ** 2) / (2.0 * sigma ** 2))
        w02 = np.exp(-(0.2 ** 2) / (2.0 * sigma ** 2))
        weighted_sum_c0 = 1.0 * lig_values[0] + w01 * lig_values[1] + w02 * lig_values[2]
        correct_value_c0 = scale_factor / N * weighted_sum_c0
        wrong_value_if_divided_by_3 = scale_factor / 3 * weighted_sum_c0

        assert np.isclose(sparse.loc["c0", "L1"], correct_value_c0, atol=1e-10)
        assert not np.isclose(sparse.loc["c0", "L1"], wrong_value_if_divided_by_3)


# ---------------------------------------------------------------------------
# 8. received_ligands end-to-end (per-radius secreted-vs-contact grouping)
# ---------------------------------------------------------------------------

def _narrow_ignore_eps_and_chunk(xy, lig_df, radius, scale_factor, eps=None, chunk_size=None):
    """Adapter so the narrow dense reference (which has no `eps`/`chunk_size` kwargs) can be
    monkeypatched in place of `compute_radius_weights_fast` inside `received_ligands` (which
    always passes `eps=`/`chunk_size=` by keyword)."""
    return _dense_narrow_reference(xy, lig_df, radius, scale_factor)


class TestReceivedLigandsEndToEnd:
    def _build_synthetic_inputs(self, seed=42, N=60):
        rng = np.random.default_rng(seed)
        xy = rng.uniform(0, 1000, size=(N, 2))
        obs_index = pd.Index([f"cell_{i}" for i in range(N)])
        ligand_genes = ["LigSecA", "LigSecB", "LigContactA", "LigContactB"]
        ligands_df = pd.DataFrame(
            rng.random((N, len(ligand_genes))), index=obs_index, columns=ligand_genes
        )
        # Two distinct radius groups -- mirrors init_received_ligands's secreted (large
        # radius) vs contact (small `contact_distance`) split.
        lr_info = pd.DataFrame(
            {
                "ligand": ligand_genes,
                "receptor": ["RecA", "RecB", "RecC", "RecD"],
                "pathway": ["P1", "P1", "P2", "P2"],
                "signaling": [
                    "Secreted Signaling",
                    "Secreted Signaling",
                    "Cell-Cell Contact",
                    "Cell-Cell Contact",
                ],
                "radius": [150, 150, 25, 25],
            }
        )
        return xy, ligands_df, lr_info

    def test_sparse_matches_narrow_reference_including_secreted_and_contact_grouping(self):
        xy, ligands_df, lr_info = self._build_synthetic_inputs()

        sparse_result = received_ligands(xy, ligands_df, lr_info, scale_factor=100)

        with patch(
            "SpaceTravLR.models.parallel_estimators.compute_radius_weights_fast",
            side_effect=_narrow_ignore_eps_and_chunk,
        ):
            ref_result = received_ligands(xy, ligands_df, lr_info, scale_factor=100)

        assert sparse_result.shape == ref_result.shape
        assert sparse_result.index.equals(ref_result.index)
        assert list(sparse_result.columns) == list(ref_result.columns)
        assert list(sparse_result.columns) == list(ligands_df.columns)  # reindexed
        assert not sparse_result.isna().any().any()
        np.testing.assert_allclose(
            sparse_result.values, ref_result.values, rtol=1e-6, atol=1e-8
        )

    def test_duplicated_ligand_rows_and_reindex_fillna_behavior_preserved(self):
        # lr_info with a duplicated ligand row (received_ligands drop_duplicates(keep="first"))
        # and a ligand column with no lr_info entry at all (should end up all-zero via
        # reindex(...).fillna(0), exactly like the reference path).
        xy, ligands_df, lr_info = self._build_synthetic_inputs(seed=7, N=40)
        ligands_df = ligands_df.copy()
        ligands_df["OrphanLig"] = np.random.default_rng(1).random(len(ligands_df))

        dup_row = lr_info.iloc[[0]].copy()
        lr_info_with_dup = pd.concat([lr_info, dup_row], ignore_index=True)

        sparse_result = received_ligands(xy, ligands_df, lr_info_with_dup, scale_factor=50)
        with patch(
            "SpaceTravLR.models.parallel_estimators.compute_radius_weights_fast",
            side_effect=_narrow_ignore_eps_and_chunk,
        ):
            ref_result = received_ligands(xy, ligands_df, lr_info_with_dup, scale_factor=50)

        assert "OrphanLig" in sparse_result.columns
        assert np.allclose(sparse_result["OrphanLig"].values, 0.0)
        np.testing.assert_allclose(
            sparse_result.values, ref_result.values, rtol=1e-6, atol=1e-8
        )

    def test_received_ligands_chunking_does_not_change_result(self):
        """End-to-end (multi-radius-group) chunking invariance."""
        xy, ligands_df, lr_info = self._build_synthetic_inputs(seed=3, N=90)
        full = received_ligands(xy, ligands_df, lr_info, scale_factor=20, chunk_size=90)
        for cs in [1, 5, 40, 1000]:
            chunked = received_ligands(xy, ligands_df, lr_info, scale_factor=20, chunk_size=cs)
            assert np.array_equal(full.values, chunked.values), f"chunk_size={cs} diverged"


if __name__ == "__main__":
    import unittest

    # Allow `python tests/test_sparse_equivalence.py` in addition to pytest.
    pytest.main([__file__, "-v"])
