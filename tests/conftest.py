"""Shared pytest fixtures / test-process shims.

Idempotent pyarrow extension-type registration
-----------------------------------------------
Several tests mock `sys.modules`/`importlib.import_module` (e.g. the
`run_celloracle_` tests). When those patches unwind, pyarrow re-runs its lazy
registration of the pandas extension types (`pandas.period`, `pandas.interval`),
which raises `ArrowKeyError: A type extension with name pandas.period already
defined` on the *next* real `to_parquet`/`read_parquet` call in the same
process. This is a pure test-ordering artifact -- registering an identical
extension type twice is a harmless no-op -- and does not occur in the real
pipeline. We make re-registration idempotent for the whole test process so tests
that do real parquet I/O don't spuriously fail depending on run order.

Harreman drop-in test infra (CU-A, shared for CU-B/C)
------------------------------------------------------
`metab_processing/Harreman/cell_communication_lowmem.py` does `import harreman` and
resolves internal helpers off it at runtime -- but real harreman only installs on Savio
(see DataForClaude/documentation/05_harreman_reference.md). To exercise these drop-ins
locally we ship a fake `harreman` package under `tests/fixtures/fake_harreman/` (NOT the
real thing -- see its docstring). We put both the Harreman scripts dir and the fake
package's containing dir on `sys.path` here so any test can do
`import cell_communication_lowmem` / `import harreman` without repeating the plumbing.
"""
import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_DIR.parent

for _p in (
    _REPO_ROOT / "metab_processing" / "Harreman",   # `cell_communication_lowmem`, `nbhd_scores`
    _TESTS_DIR / "fixtures",                          # fixtures package namespace
    _TESTS_DIR / "fixtures" / "fake_harreman",        # the fake `harreman` package itself
):
    _p_str = str(_p)
    if _p.is_dir() and _p_str not in sys.path:
        sys.path.insert(0, _p_str)

import pyarrow as pa

_orig_register = pa.register_extension_type


def _idempotent_register(ext_type):
    try:
        return _orig_register(ext_type)
    except pa.lib.ArrowKeyError:
        # already registered (same type) -> tolerate
        return None


if getattr(pa.register_extension_type, "__name__", "") != "_idempotent_register":
    pa.register_extension_type = _idempotent_register
