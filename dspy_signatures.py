"""Top-level shim re-exporting backend.dspy_signatures for legacy imports."""

from backend.dspy_signatures import *  # noqa: F401,F403

__all__ = [name for name in dir() if not name.startswith("_")]
