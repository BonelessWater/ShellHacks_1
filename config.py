"""Safe top-level config shim used in tests.

We avoid importing `backend.config` at module import time because that module
creates an APIKeyManager instance which expects environment variables and may
raise during test collection. Instead provide a minimal, safe `api_key_manager`
implementation that exposes the same methods used by the code under test.

When available, some constants are imported from backend.config; failures are
ignored so tests can run in isolated environments.
"""

from typing import Optional


class _StubAPIKeyManager:
	def __init__(self):
		# Minimal attributes to mimic the real APIKeyManager interface
		self.api_keys = []
		self.failed_keys = set()
		self.current_key_index = 0
		self._failed = self.failed_keys

	def get_current_key(self) -> Optional[str]:
		return None

	def get_available_count(self) -> int:
		"""Return number of available (non-failed) keys. Stub returns 0."""
		return max(0, len(self.api_keys) - len(self.failed_keys))

	def rotate_key(self) -> Optional[str]:
		"""Rotate to next key in the stub (no-op)."""
		if not self.api_keys:
			return None
		self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
		return None

	def mark_key_failed(self, key: str, error_msg: str = ""):
		self._failed.add(key)

	def reset_failures(self):
		self._failed.clear()


# Provide a minimal api_key_manager instance for tests
api_key_manager = _StubAPIKeyManager()

# Try to import real constants from backend.config if available, but do not
# allow import-time failures to crash test collection.
try:
	from backend.archive.config import (
		APPROVED_VENDORS,
		SUSPICIOUS_KEYWORDS,
		HIGH_VALUE_THRESHOLD,
		HIGH_QUANTITY_THRESHOLD,
		TOTAL_MISMATCH_TOLERANCE,
	)
except Exception:
	APPROVED_VENDORS = set()
	SUSPICIOUS_KEYWORDS = []
	HIGH_VALUE_THRESHOLD = 1000.0
	HIGH_QUANTITY_THRESHOLD = 100
	TOTAL_MISMATCH_TOLERANCE = 0.01

__all__ = [
	"api_key_manager",
	"APPROVED_VENDORS",
	"SUSPICIOUS_KEYWORDS",
	"HIGH_VALUE_THRESHOLD",
	"HIGH_QUANTITY_THRESHOLD",
	"TOTAL_MISMATCH_TOLERANCE",
]
