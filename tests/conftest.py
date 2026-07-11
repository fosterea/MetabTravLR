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
"""
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
