"""Minimal fake ``harreman`` package for local (non-Savio) tests.

harreman itself only installs/runs on Savio (heavy deps: torch+CUDA stack, custom
Hotspot vendoring, etc. -- see DataForClaude/documentation/05_harreman_reference.md).
This package is NOT harreman -- it exists purely so `metab_processing/Harreman`'s
low-memory drop-ins (which do ``import harreman`` and resolve helpers via
``inspect.getmodule(harreman.tools.compute_interacting_cell_scores)`` at runtime) can be
exercised locally, and so tests can assert a drop-in's outputs match the *stock
algorithm* on tiny synthetic data.

``harreman.tools`` vendors the real stock functions verbatim -- both from
``DataForClaude/cell_communication.py`` (our read-only reference copy of harreman's
source, confirmed byte-identical to the real repo) and, for the three helpers that file
imports from other harreman submodules (``counts_from_anndata``, ``standardize_counts``,
``make_weights_non_redundant``), from a locally-cloned copy of the real harreman source.
See ``tools.py``'s module docstring for exact provenance / line numbers.
"""
from . import tools

__all__ = ["tools"]
