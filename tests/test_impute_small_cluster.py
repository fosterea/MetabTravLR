"""Tier-1: `BaseTravLR.impute_clusterwise` falls back to raw counts for clusters below
`min_cells_for_magic` (16 by default) instead of crashing MAGIC's kNN graph
(n_neighbors=16 > n_samples). A large cluster still gets imputed (values change).
Requires the `magic` package (model env), not the pure Tier-0 loop.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np
import pandas as pd
import anndata as ad

from SpaceTravLR.oracles import BaseTravLR


def _make_adata(small_n=6, big_n=24, n_genes=20, seed=0):
    rng = np.random.default_rng(seed)
    n = small_n + big_n
    X = rng.random((n, n_genes)).astype(np.float32)
    a = ad.AnnData(X=X)
    a.var_names = [f"g{i}" for i in range(n_genes)]
    a.obs_names = [f"c{i}" for i in range(n)]
    a.obs["cell_type"] = pd.Categorical(["small"] * small_n + ["big"] * big_n)
    a.layers["normalized_count"] = X.copy()
    return a


def test_small_cluster_passes_raw_through():
    adata = _make_adata(small_n=6, big_n=24)
    BaseTravLR.impute_clusterwise(adata)  # default min_cells_for_magic=16; must not raise

    assert "imputed_count" in adata.layers
    imputed = pd.DataFrame(adata.layers["imputed_count"], index=adata.obs_names, columns=adata.var_names)
    raw = pd.DataFrame(adata.layers["normalized_count"], index=adata.obs_names, columns=adata.var_names)

    small = adata.obs["cell_type"] == "small"
    # sub-threshold cluster: imputed == raw (no MAGIC)
    np.testing.assert_allclose(imputed[small].values, raw[small].values)
    # normal cluster: MAGIC actually changed the values
    assert not np.allclose(imputed[~small].values, raw[~small].values)


def test_threshold_is_configurable():
    adata = _make_adata(small_n=6, big_n=24)
    # raise the threshold so BOTH clusters fall below it -> whole layer is raw passthrough
    BaseTravLR.impute_clusterwise(adata, min_cells_for_magic=100)
    np.testing.assert_allclose(adata.layers["imputed_count"], adata.layers["normalized_count"])


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
