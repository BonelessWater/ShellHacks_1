"""Shim to expose archived data models as `backend.data_models`.

This module re-exports public symbols from `backend.archive.data_models` so
existing imports continue to work during/after a large refactor. It keeps
the compatibility surface small and is safe to remove once callers are
updated.
"""

from backend import compat

# Re-export public symbols from backend.archive.data_models (or fallbacks)
compat.reexport_module("data_models", globals())
