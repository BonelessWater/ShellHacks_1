"""Top-level shim re-exporting backend.agents for legacy imports.

Some modules import `agents` as a top-level module. Re-export the common
symbols from `backend.agents` so those imports succeed during tests.
"""

from backend.agents import *  # noqa: F401,F403

__all__ = [name for name in dir() if not name.startswith("_")]
