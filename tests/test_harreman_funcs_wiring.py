"""Wiring/regression guard for `metab_processing/Harreman/harreman_funcs.py` (CU-D).

`harreman_funcs.py` imports the real `harreman` (Savio-only), so we can't import the
module here -- instead we (1) statically parse it to confirm the two aggregate CCC calls
go through the memory-safe drop-ins (not the OOM-prone stock `harreman.tl.*`), and
(2) bind the exact keyword args each call site passes against the drop-in signatures, so a
future signature change that would break the real pipeline fails loudly in CI.

Why this matters: the whole point of CU-B/C/D is that `HarremanRunner.run_harreman` stops
OOMing at Xenium scale. A silent revert to `harreman.tl.compute_cell_communication` (or a
kwarg the drop-in no longer accepts) would only surface on an expensive Savio run.
"""
import ast
import inspect
import unittest
from pathlib import Path

import harreman  # noqa: F401  (fake package via conftest; ensures the import path is set up)
from cell_communication_lowmem import (
    compute_cell_communication_lowmem,
    compute_ct_cell_communication_lowmem,
)

HARREMAN_FUNCS = Path(__file__).resolve().parents[1] / "metab_processing" / "Harreman" / "harreman_funcs.py"

# lowmem drop-in name -> (stock name it replaces, drop-in callable)
WIRING = {
    "compute_cell_communication_lowmem": ("compute_cell_communication", compute_cell_communication_lowmem),
    "compute_ct_cell_communication_lowmem": ("compute_ct_cell_communication", compute_ct_cell_communication_lowmem),
}


def _call_name(node):
    """Return the callable's simple name for a Call node (`f(...)` -> 'f', `a.b.f(...)` -> 'f')."""
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


class HarremanFuncsWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(HARREMAN_FUNCS.read_text())
        cls.calls = [n for n in ast.walk(cls.tree) if isinstance(n, ast.Call)]

    def test_uses_lowmem_dropins_not_stock_aggregate_calls(self):
        called = {_call_name(c) for c in self.calls}
        for lowmem_name, (stock_name, _) in WIRING.items():
            self.assertIn(lowmem_name, called, f"harreman_funcs.py should call {lowmem_name}")
            self.assertNotIn(
                stock_name, called,
                f"harreman_funcs.py still calls stock {stock_name} (OOM-prone) -- must use {lowmem_name}",
            )

    def test_call_site_kwargs_bind_to_dropin_signatures(self):
        """Every kwarg each call site passes must be accepted by the drop-in signature."""
        for lowmem_name, (_, fn) in WIRING.items():
            sig = inspect.signature(fn)
            call_nodes = [c for c in self.calls if _call_name(c) == lowmem_name]
            self.assertTrue(call_nodes, f"no call to {lowmem_name} found")
            for c in call_nodes:
                kwargs = {kw.arg: None for kw in c.keywords if kw.arg is not None}
                n_positional = len(c.args)  # adata is passed positionally
                with self.subTest(call=lowmem_name):
                    # bind positional placeholders + the kwarg names; raises TypeError on drift
                    sig.bind_partial(*([None] * n_positional), **kwargs)

    def test_chunk_size_is_threaded(self):
        """Both call sites must pass gene_pair_chunk_size (the memory knob), else the
        adaptive default silently never reaches the drop-in from HarremanRunner."""
        for lowmem_name in WIRING:
            call_nodes = [c for c in self.calls if _call_name(c) == lowmem_name]
            for c in call_nodes:
                kw_names = {kw.arg for kw in c.keywords}
                self.assertIn("gene_pair_chunk_size", kw_names, f"{lowmem_name} call missing gene_pair_chunk_size")


if __name__ == "__main__":
    unittest.main()
