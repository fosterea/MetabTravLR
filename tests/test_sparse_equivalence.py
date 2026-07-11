"""Correctness tests for the O(N^2) -> sparse/KDTree swap in
`SpaceTravLR.models.parallel_estimators`.

Two operations were made sparse:

1. `create_spatial_features` (per-cell-type neighbor COUNT within a radius): replaced the
   dense `cdist(coords, coords) <= radius` reference with a `cKDTree` radius query. This is
   a pure counting operation on a boolean `<= radius` mask, so the sparse version must match
   the dense reference **exactly** (`np.array_equal`), not just approximately. The dense
   reference implementation is kept as `_create_spatial_features_dense` for this purpose.

2. `received_ligands` / `compute_radius_weights_fast` (Gaussian-kernel-weighted ligand
   sum): replaced the dense N x N Gaussian weight matrix with a `cKDTree`
   `sparse_distance_matrix` radius query truncated at a cutoff `C = radius*sqrt(2*ln(1/eps))`
   chosen so every dropped term has weight < eps (default eps=1e-9, C ~= 6.4*radius). This is
   an approximation (by construction), so equivalence is checked with
   `np.allclose(rtol=1e-6, atol=1e-8)`, not exact equality. The dense reference
   implementation is kept as `compute_radius_weights_fast_dense`.

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

from SpaceTravLR.models.parallel_estimators import (
    compute_radius_weights_fast_dense,
    compute_radius_weights_fast_sparse,
    create_spatial_features,
    received_ligands,
    _create_spatial_features_dense,
)

RNG_SEED = 12345


def _clustered_coords(n, n_clusters=4, spread=5.0, extent=1000.0, seed=0):
    """A handful of tight Gaussian blobs, as a stand-in for real tissue spatial structure
    (as opposed to the uniform-random coordinates used elsewhere)."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0, extent, size=(n_clusters, 2))
    assignments = rng.integers(0, n_clusters, size=n)
    return centers[assignments] + rng.normal(0, spread, size=(n, 2))


# ---------------------------------------------------------------------------
# 1. create_spatial_features: EXACT equivalence (counting op, no tolerance needed)
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
# 2. received-ligand kernel: tight-tolerance equivalence (approximation by construction)
# ---------------------------------------------------------------------------

class TestReceivedLigandKernelEquivalence:
    def test_dense_vs_sparse_allclose_across_many_configurations(self):
        rng = np.random.default_rng(RNG_SEED + 1)
        max_abs_diff = 0.0
        n_configs = 0
        for N in [30, 150, 600]:
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

                            dense = compute_radius_weights_fast_dense(
                                xy, lig, radius, scale_factor
                            )
                            sparse = compute_radius_weights_fast_sparse(
                                xy, lig, radius, scale_factor
                            )

                            assert dense.index.equals(sparse.index)
                            assert list(dense.columns) == list(sparse.columns)
                            assert dense.shape == sparse.shape

                            diff = np.abs(dense.values - sparse.values)
                            max_abs_diff = max(max_abs_diff, float(diff.max()))
                            assert np.allclose(
                                dense.values, sparse.values, rtol=1e-6, atol=1e-8
                            )
                            n_configs += 1
        assert n_configs >= 50
        # Sanity print (visible with `pytest -s`) -- worst-case error observed in the sweep.
        print(f"\n[test_sparse_equivalence] max |dense - sparse| over sweep = {max_abs_diff:.3e}")

    def test_equivalence_holds_at_large_magnitude(self):
        """Guards that equivalence is governed by RELATIVE tolerance, not the absolute
        atol used in the sweep. Absolute error scales with scale_factor*ligand, so at
        large magnitude (values ~1e9 here) absolute diffs are ~1e-2 -- far above atol=1e-8
        -- yet the sparse result still matches the dense reference to rtol=1e-6."""
        rng = np.random.default_rng(RNG_SEED + 7)
        xy = _clustered_coords(400, n_clusters=5, spread=10.0, seed=123)
        lig = pd.DataFrame(
            rng.uniform(500, 1500, size=(400, 4)),
            index=[f"cell_{i}" for i in range(400)],
            columns=[f"Lig{i}" for i in range(4)],
        )
        radius, scale_factor = 100.0, 1e6
        dense = compute_radius_weights_fast_dense(xy, lig, radius, scale_factor)
        sparse = compute_radius_weights_fast_sparse(xy, lig, radius, scale_factor)
        # atol=0.0 -> pure relative comparison; proves rtol is what actually protects us
        assert np.allclose(dense.values, sparse.values, rtol=1e-6, atol=0.0)
        rel = np.abs(dense.values - sparse.values) / np.maximum(np.abs(dense.values), 1e-12)
        assert rel.max() < 1e-6


# ---------------------------------------------------------------------------
# 3. Edge cases
# ---------------------------------------------------------------------------

class TestReceivedLigandEdgeCases:
    def test_single_cell(self):
        xy = np.array([[10.0, 20.0]])
        lig = pd.DataFrame({"L1": [3.0]}, index=["c0"])
        radius, sf = 5.0, 2.0
        dense = compute_radius_weights_fast_dense(xy, lig, radius, sf)
        sparse = compute_radius_weights_fast_sparse(xy, lig, radius, sf)
        expected = sf / 1 * 3.0  # only self, weight 1, N=1
        assert np.isclose(dense.loc["c0", "L1"], expected)
        assert np.isclose(sparse.loc["c0", "L1"], expected, atol=1e-12)

    def test_all_cells_at_same_location_everyone_is_a_neighbor(self):
        N = 6
        xy = np.zeros((N, 2))
        lig_values = np.arange(1, N + 1, dtype=float)
        lig = pd.DataFrame({"L1": lig_values}, index=[f"c{i}" for i in range(N)])
        radius, sf = 10.0, 1.0
        dense = compute_radius_weights_fast_dense(xy, lig, radius, sf)
        sparse = compute_radius_weights_fast_sparse(xy, lig, radius, sf)
        # every pairwise distance is 0 -> every weight is 1 -> received[i] = sf/N * sum(lig)
        expected = sf / N * lig_values.sum()
        assert np.allclose(dense["L1"].values, expected)
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
        dense = compute_radius_weights_fast_dense(xy, lig, 50.0, 10.0)
        sparse = compute_radius_weights_fast_sparse(xy, lig, 50.0, 10.0)
        assert np.allclose(dense.values, 0.0)
        assert np.allclose(sparse.values, 0.0, atol=1e-12)

    def test_duplicate_coordinates(self):
        N = 10
        rng = np.random.default_rng(4)
        half = rng.uniform(0, 200, size=(N // 2, 2))
        xy = np.vstack([half, half])  # exact duplicate coordinates
        lig = pd.DataFrame({"L1": rng.random(N)}, index=[f"c{i}" for i in range(N)])
        dense = compute_radius_weights_fast_dense(xy, lig, 30.0, 5.0)
        sparse = compute_radius_weights_fast_sparse(xy, lig, 30.0, 5.0)
        assert np.allclose(dense.values, sparse.values, rtol=1e-6, atol=1e-8)


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
# 4. Determinism + the 1/N-not-1/k normalization guard
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
        radius = 1.0  # cutoff ~= 6.4 * radius, way smaller than the inter-group spacing
        scale_factor = 2.0
        lig_values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        lig = pd.DataFrame({"L1": lig_values}, index=[f"c{i}" for i in range(N)])

        sparse = compute_radius_weights_fast_sparse(xy, lig, radius, scale_factor)
        dense = compute_radius_weights_fast_dense(xy, lig, radius, scale_factor)

        # Isolated cell c3: only itself is within the cutoff (neighbor count == 1), but the
        # correct divisor is the TOTAL cell count N == 5.
        correct_value = scale_factor / N * lig_values[3]
        wrong_value_if_divided_by_neighbor_count = scale_factor / 1 * lig_values[3]

        assert np.isclose(sparse.loc["c3", "L1"], correct_value, atol=1e-12)
        assert np.isclose(dense.loc["c3", "L1"], correct_value, atol=1e-8)
        assert not np.isclose(sparse.loc["c3", "L1"], wrong_value_if_divided_by_neighbor_count)

        # Same check for cell c4.
        correct_value_c4 = scale_factor / N * lig_values[4]
        assert np.isclose(sparse.loc["c4", "L1"], correct_value_c4, atol=1e-12)

        # Clustered cell c0: neighbor count == 3 (itself + c1 + c2), but divisor must still
        # be N == 5, not 3.
        # weight(c0->c1) = exp(-0.1^2/(2*1^2)), weight(c0->c2) = exp(-0.2^2/(2*1^2))
        w01 = np.exp(-(0.1 ** 2) / (2.0 * radius ** 2))
        w02 = np.exp(-(0.2 ** 2) / (2.0 * radius ** 2))
        weighted_sum_c0 = 1.0 * lig_values[0] + w01 * lig_values[1] + w02 * lig_values[2]
        correct_value_c0 = scale_factor / N * weighted_sum_c0
        wrong_value_if_divided_by_3 = scale_factor / 3 * weighted_sum_c0

        assert np.isclose(sparse.loc["c0", "L1"], correct_value_c0, atol=1e-10)
        assert not np.isclose(sparse.loc["c0", "L1"], wrong_value_if_divided_by_3)


# ---------------------------------------------------------------------------
# 5. received_ligands end-to-end (per-radius secreted-vs-contact grouping)
# ---------------------------------------------------------------------------

def _dense_ignore_eps(xy, lig_df, radius, scale_factor, eps=None):
    """Adapter so the dense reference (which has no `eps` kwarg) can be monkeypatched in
    place of `compute_radius_weights_fast` inside `received_ligands` (which always passes
    `eps=` by keyword)."""
    return compute_radius_weights_fast_dense(xy, lig_df, radius, scale_factor)


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

    def test_sparse_matches_dense_including_secreted_and_contact_grouping(self):
        xy, ligands_df, lr_info = self._build_synthetic_inputs()

        sparse_result = received_ligands(xy, ligands_df, lr_info, scale_factor=100)

        with patch(
            "SpaceTravLR.models.parallel_estimators.compute_radius_weights_fast",
            side_effect=_dense_ignore_eps,
        ):
            dense_result = received_ligands(xy, ligands_df, lr_info, scale_factor=100)

        assert sparse_result.shape == dense_result.shape
        assert sparse_result.index.equals(dense_result.index)
        assert list(sparse_result.columns) == list(dense_result.columns)
        assert list(sparse_result.columns) == list(ligands_df.columns)  # reindexed
        assert not sparse_result.isna().any().any()
        np.testing.assert_allclose(
            sparse_result.values, dense_result.values, rtol=1e-6, atol=1e-8
        )

    def test_duplicated_ligand_rows_and_reindex_fillna_behavior_preserved(self):
        # lr_info with a duplicated ligand row (received_ligands drop_duplicates(keep="first"))
        # and a ligand column with no lr_info entry at all (should end up all-zero via
        # reindex(...).fillna(0), exactly like the dense path).
        xy, ligands_df, lr_info = self._build_synthetic_inputs(seed=7, N=40)
        ligands_df = ligands_df.copy()
        ligands_df["OrphanLig"] = np.random.default_rng(1).random(len(ligands_df))

        dup_row = lr_info.iloc[[0]].copy()
        lr_info_with_dup = pd.concat([lr_info, dup_row], ignore_index=True)

        sparse_result = received_ligands(xy, ligands_df, lr_info_with_dup, scale_factor=50)
        with patch(
            "SpaceTravLR.models.parallel_estimators.compute_radius_weights_fast",
            side_effect=_dense_ignore_eps,
        ):
            dense_result = received_ligands(xy, ligands_df, lr_info_with_dup, scale_factor=50)

        assert "OrphanLig" in sparse_result.columns
        assert np.allclose(sparse_result["OrphanLig"].values, 0.0)
        np.testing.assert_allclose(
            sparse_result.values, dense_result.values, rtol=1e-6, atol=1e-8
        )


if __name__ == "__main__":
    import unittest

    # Allow `python tests/test_sparse_equivalence.py` in addition to pytest.
    pytest.main([__file__, "-v"])
