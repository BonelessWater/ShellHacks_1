"""Compatibility shim for backend.dspy_signatures re-exporting archived code."""

"""Shim for `backend.dspy_signatures` (temporary compatibility layer).

Exports public symbols from `backend.archive.dspy_signatures`. Remove after
cleanup PR once callers are aligned.
"""

from backend import compat

# Re-export public symbols from backend.archive.dspy_signatures (or fallbacks)
compat.reexport_module("dspy_signatures", globals())
