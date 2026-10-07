"""Tier-0/1 tests for `metab_processing/LinearRegression/hvg_prep.py`'s resilience to
Savio's intermittent Lustre errors (errno 5 EIO on writes, errno 108 ESHUTDOWN /
BrokenPipeError on reads/stats -- both surface as `OSError`, and `BrokenPipeError` is a
subclass of it).

Covers:
  - `_run_with_retry` retries on `OSError` (incl. `BrokenPipeError`) with linear backoff,
    gives up after `retries` attempts, and does NOT retry a non-`OSError` exception.
  - `run`'s up-front skip-if-already-present check (resumability): a valid, readable
    existing output short-circuits before any read/build work; a missing output does not.

No heavy env needed for the retry tests (they monkeypatch `hp.run` directly, never
touching scanpy/torch). The skip-existing tests monkeypatch `hp.build_x_adata` and
`hp.sc.read_h5ad` so `run()` can execute without real data; the "written" h5ad in the
positive case is a real (if minimal) file via `anndata`, matching the module's own
`_h5ad_is_readable` check (`h5py.File` open + `'var'`/`'obs'` present).
"""
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import metab_processing.LinearRegression.hvg_prep as hp


def write_tiny_h5ad(path):
    """A real, minimal, readable h5ad -- satisfies `_h5ad_is_readable` (has 'var'/'obs')."""
    import anndata as ad
    import numpy as np

    adata = ad.AnnData(np.zeros((2, 2), dtype="float32"))
    adata.var_names = ["A", "B"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path)


class RetryTests(unittest.TestCase):
    def test_retry_then_succeed(self):
        calls = {"n": 0}

        def flaky(path, top_n, annot):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise OSError("boom")

        with mock.patch.object(hp, "run", side_effect=flaky):
            ok = hp._run_with_retry("/some/path", "all", "annot", retries=3, wait=0)

        self.assertTrue(ok)
        self.assertEqual(calls["n"], 3)

    def test_gives_up_after_retries(self):
        with mock.patch.object(hp, "run", side_effect=OSError("boom")) as m:
            ok = hp._run_with_retry("/some/path", "all", "annot", retries=2, wait=0)

        self.assertFalse(ok)
        self.assertEqual(m.call_count, 2)

    def test_non_retryable_not_retried(self):
        with mock.patch.object(hp, "run", side_effect=ValueError("not raw counts")) as m:
            ok = hp._run_with_retry("/some/path", "all", "annot", retries=3, wait=0)

        self.assertFalse(ok)
        self.assertEqual(m.call_count, 1)

    def test_broken_pipe_error_is_retried(self):
        # BrokenPipeError is a subclass of OSError (ESHUTDOWN on reads/stats).
        calls = {"n": 0}

        def flaky(path, top_n, annot):
            calls["n"] += 1
            if calls["n"] == 1:
                raise BrokenPipeError("pipe broke")

        with mock.patch.object(hp, "run", side_effect=flaky):
            ok = hp._run_with_retry("/some/path", "all", "annot", retries=3, wait=0)

        self.assertTrue(ok)
        self.assertEqual(calls["n"], 2)


class SkipExistingTests(unittest.TestCase):
    def test_skip_when_output_already_readable(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp) / "dataset"
            out = dataset_dir / "LinearRegression" / "hvgall_x_adata.h5ad"
            write_tiny_h5ad(out)

            build_calls = {"n": 0}

            def fake_build_x_adata(*args, **kwargs):
                build_calls["n"] += 1

            with mock.patch.object(hp, "build_x_adata", side_effect=fake_build_x_adata), \
                 mock.patch.object(hp.sc, "read_h5ad", side_effect=AssertionError(
                     "should not read adata.h5ad when output already present")):
                hp.run(dataset_dir, "all", "annot")

            self.assertEqual(build_calls["n"], 0)

    def test_no_skip_when_output_missing(self):
        import tempfile
        import numpy as np
        import scipy.sparse as sp
        import anndata as ad

        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp) / "dataset"
            dataset_dir.mkdir(parents=True)
            # No LinearRegression/hvgall_x_adata.h5ad yet -> skip must be bypassed.

            # Sparse (as a real h5ad's counts matrix is) -- a dense ndarray's `.data` is a
            # raw-buffer memoryview, not the nonzero values, which trips the module's
            # pre-existing raw-counts check unrelated to what this test covers.
            stub_adata = ad.AnnData(sp.csr_matrix(np.array([[1.0, 2.0], [0.0, 3.0]], dtype="float32")))
            stub_adata.var_names = ["A", "B"]

            build_calls = {"n": 0}

            def fake_build_x_adata(*args, **kwargs):
                build_calls["n"] += 1

            with mock.patch.object(hp.sc, "read_h5ad", return_value=stub_adata), \
                 mock.patch.object(hp, "build_x_adata", side_effect=fake_build_x_adata):
                hp.run(dataset_dir, "all", "annot")

            self.assertEqual(build_calls["n"], 1)


if __name__ == "__main__":
    unittest.main()
