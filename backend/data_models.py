"""Shim to expose archived data models as `backend.data_models`.

Some modules import `backend.data_models` directly; during the merge the
authoritative implementations live under `backend.archive.data_models`.
This shim re-exports the archived implementations to preserve backward
compatibility for tests and other imports.
"""

"""Temporary shim for `backend.data_models`.

This module re-exports public symbols from `backend.archive.data_models` so
existing imports continue to work during/after a large merge. It's a
short-term compatibility layer; see `backend/compat.py` for the re-export
implementation and add a cleanup PR to remove this shim once callers are
updated.
"""

from backend import compat

# Re-export public symbols from backend.archive.data_models (or fallbacks)
compat.reexport_module("data_models", globals())
"""Shim to expose archived data models as `backend.data_models`.

This file intentionally re-exports the canonical implementations from
`backend.archive.data_models`. It used to contain duplicate definitions; this
cleanup reduces it to a single re-export so callers keep working while the
archive module remains authoritative.
"""

from backend import compat

# Re-export public symbols from backend.archive.data_models
compat.reexport_module("data_models", globals())
@dataclass
