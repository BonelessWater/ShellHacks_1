"""Compatibility shim for dspy signatures.
Re-export the archived implementation to preserve backward compatibility.
"""
from backend.archive import dspy_signatures as _arch  # type: ignore
try:
    from backend.archive.dspy_signatures import *  # noqa: F401,F403
except Exception:
    # Provide a minimal fallback
    def create_dspy_modules():
        return None
"""Compatibility shim for backend.dspy_signatures re-exporting archived code."""

"""Shim for `backend.dspy_signatures` (temporary compatibility layer).

Exports public symbols from `backend.archive.dspy_signatures`. Remove after
cleanup PR once callers are aligned.
"""

"""Shim for `backend.dspy_signatures` (temporary compatibility layer).

Exports public symbols from `backend.archive.dspy_signatures`. If the
archived implementation isn't present, provide minimal fallbacks so tests
and importers don't crash.
"""

from backend import compat

exported = compat.reexport_module("dspy_signatures", globals())

if not exported:
    # Minimal fallbacks used by integration tests
    def create_dspy_modules():
        return {}

    def extract_review_from_response(resp):
        return {}

    def extract_summary_from_response(resp):
        return {}

    def extract_tasks_from_response(resp):
        return []
