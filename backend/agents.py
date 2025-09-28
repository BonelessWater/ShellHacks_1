"""Compatibility shim exporting the archived agents implementation.

This keeps `backend.agents` imports working after the merge that moved the
implementation into `backend.archive`.
"""

"""Shim for `backend.agents` that re-exports archived implementation.

See `backend/compat.py` for the re-export helper. This file is temporary and
can be removed in a follow-up cleanup PR once callers are migrated.
"""

from backend import compat

# Re-export public symbols from backend.archive.agents (or fallbacks)
compat.reexport_module("agents", globals())
