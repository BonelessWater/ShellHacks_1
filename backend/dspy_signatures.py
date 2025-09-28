"""Compatibility shim for backend.dspy_signatures re-exporting archived code."""

from backend import compat

# Re-export public symbols from backend.archive.dspy_signatures (or fallbacks)
compat.reexport_module("dspy_signatures", globals())
