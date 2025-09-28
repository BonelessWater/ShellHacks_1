"""Compatibility shim for backend.main_pipeline

The authoritative implementation was moved to `backend.archive.main_pipeline` during
merge; many tests and modules still import `backend.main_pipeline`. Re-export the
archive implementation here to maintain compatibility.
"""

"""Shim for `backend.main_pipeline` to re-export archived implementation.

This is a temporary shim. See `backend/compat.py` and create a cleanup PR to
remove these shims after refactoring call sites.
"""

from backend import compat

# Re-export public symbols from backend.archive.main_pipeline (or fallbacks)
compat.reexport_module("main_pipeline", globals())
