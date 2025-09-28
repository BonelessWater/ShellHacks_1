"""Compatibility shim exporting the archived agents implementation.

This keeps `backend.agents` imports working after the merge that moved the
implementation into `backend.archive`.
"""

from backend import compat

# Re-export public symbols from backend.archive.agents (or fallbacks)
compat.reexport_module("agents", globals())
