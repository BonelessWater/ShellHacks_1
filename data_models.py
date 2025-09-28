"""Top-level shim for backend.data_models used by legacy imports in backend modules and tests."""

from backend.data_models import *  # noqa: F401,F403

__all__ = [name for name in dir() if not name.startswith("_")]
