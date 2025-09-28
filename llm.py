"""Lightweight LLm shim for tests.

The real `backend.llm` integrates with external LLM providers. For tests we
expose a safe `get_configured_lm` function that returns None so code paths that
try to initialize an LLM will detect it's unavailable and fallback to
non-LLM behavior.
"""

from typing import Any, Optional


def get_configured_lm() -> Optional[Any]:
	"""Return None to indicate no LLM is configured in test environments."""
	return None

__all__ = ["get_configured_lm"]
