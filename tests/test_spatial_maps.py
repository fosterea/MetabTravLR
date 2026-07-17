import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from sklearn.datasets import make_regression
from unittest import TestCase
import numpy as np
import pandas as pd

from SpaceTravLR.models.spatial_map import xyc2spatial, xyc2spatial_fast
from SpaceTravLR.models.parallel_estimators import create_spatial_features

class SpatialMapsTest(TestCase):

    # def test_xyc2spatial(self):
    #     n_samples = np.random.randint(100, 1000)
    #     n_clusters = np.random.randint(2, 10)
    #     m = n = np.random.randint(3, 12)
        
    #     X, y = make_regression(n_samples=n_samples, n_features=2, noise=0.1)
    #     labels = np.random.randint(0, n_clusters, (n_samples,))

    #     self.assertEqual(
    #         xyc2spatial(X[:, 0], X[:, 1], labels, m, n).shape, 
    #         (n_samples, n_clusters, m, n)
    #     )

    def test_xyc2spatial_fast(self):
        # NOTE: `xyc2spatial_fast` now returns a single channel `(N, 1, m, n)`
        # instead of one (bit-identical) channel per cluster `(N, n_clusters, m, n)`.
        # All channels used to be exact copies of each other (the per-cluster
        # mask was a no-op), so storing n_clusters copies was pure wasted
        # memory; see `models/spatial_map.py::xyc2spatial_fast` and
        # `tests/test_spatial_maps_single_channel.py` for the bit-identity
        # proof against the old multi-channel behavior. This test now only
        # asserts the (updated) shape contract.
        n_samples = np.random.randint(100, 1000)
        n_clusters = np.random.randint(2, 10)
        m = n = np.random.randint(3, 12)

        X, y = make_regression(n_samples=n_samples, n_features=2, noise=0.1)
        labels = np.random.randint(0, n_clusters, (n_samples,))

        spatial_maps = xyc2spatial_fast(
            xyc = np.column_stack([X, labels]),
            m=m,
            n=n,
        ).astype(np.float32)

        self.assertEqual(
            spatial_maps.shape,
            (n_samples, 1, m, n)
        )
    
    def test_create_spatial_features(self):
        # Create test data
        n_samples = 100
        n_clusters = 3
        radius = 50
        
        # Generate random coordinates
        x = np.random.uniform(0, 1000, n_samples)
        y = np.random.uniform(0, 1000, n_samples)
        
        # Generate random cell types
        celltypes = np.random.randint(0, n_clusters, n_samples)
        unique_celltypes = np.unique(celltypes)
        
        # Create index
        obs_index = pd.Index([f'cell_{i}' for i in range(n_samples)])
        
        # Call the function
        result = create_spatial_features(x, y, celltypes, obs_index, radius=radius)
        
        # Test the shape of the result
        self.assertEqual(result.shape, (n_samples, len(unique_celltypes)))
        
        # Test that column names are correctly formatted
        expected_columns = [f'{ct}_within' for ct in unique_celltypes]
        self.assertListEqual(list(result.columns), expected_columns)
        
        # Test that the index is preserved
        self.assertTrue(result.index.equals(obs_index))
        
        # Test that values are non-negative integers
        self.assertTrue(np.all(result.values >= 0))
        
        # Test a specific case with known outcome
        # Create a small dataset with predictable distances
        x_small = np.array([0, 0, 100])
        y_small = np.array([0, 40, 0])
        celltypes_small = np.array([0, 0, 1])
        obs_index_small = pd.Index(['cell_1', 'cell_2', 'cell_3'])
        
        # With radius 50, the first two cells should see each other as type 0
        # The third cell should see no cells of type 1 within radius
        result_small = create_spatial_features(x_small, y_small, celltypes_small, 
                                              obs_index_small, radius=50)
        
        # First cell should see 1 cell of type 0 (itself) and 0 of type 1
        self.assertEqual(result_small.iloc[0, 0], 2)  # 2 cells of type 0 (itself + cell_2)
        self.assertEqual(result_small.iloc[0, 1], 0)  # 0 cells of type 1
        
        # Second cell should see 1 cell of type 0 (itself) and 0 of type 1
        self.assertEqual(result_small.iloc[1, 0], 2)  # 2 cells of type 0 (itself + cell_1)
        self.assertEqual(result_small.iloc[1, 1], 0)  # 0 cells of type 1
        
        # Third cell should see 0 cells of type 0 and 1 cell of type 1 (itself)
        self.assertEqual(result_small.iloc[2, 0], 0)  # 0 cells of type 0
        self.assertEqual(result_small.iloc[2, 1], 1)  # 1 cell of type 1 (itself)

if __name__ == '__main__':
    import unittest
    unittest.main(verbosity=2)
        