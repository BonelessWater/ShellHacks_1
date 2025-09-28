"""Compatibility utilities for backend modules.

This module provides a small helper used by shim files under `backend/` to
re-export public symbols from implementations that moved to
`backend.archive.*`. It's intentionally conservative: if the archived
implementation isn't present, it will fall back to a top-level module if
available, otherwise it returns an empty list and the shim should provide
its own fallbacks.
"""

from importlib import import_module
from types import ModuleType
from typing import Dict, List


def _try_import(module_path: str) -> ModuleType | None:
    try:
        return import_module(module_path)
    except Exception:
        return None


def reexport_module(module_name: str, target_globals: Dict[str, object]) -> List[str]:
    """Copy public symbols from an implementation module into the shim's
    globals.

    Order of attempts:
    1. Try `backend.archive.<module_name>` (canonical archived location).
    2. Try `backend.<module_name>` (legacy in-package implementation).
    3. Try top-level `<module_name>` (project-level shim like `data_models`).

    Returns the list of exported names. If nothing was imported, returns
    an empty list.
    """
    candidates = [f"backend.archive.{module_name}", f"backend.{module_name}", module_name]

    mod = None
    for candidate in candidates:
        mod = _try_import(candidate)
        if mod is not None:
            break

    if mod is None:
        return []

    names = getattr(mod, "__all__", None)
    if names is None:
        names = [n for n in dir(mod) if not n.startswith("_")]

    exported: List[str] = []
    for name in names:
        if name in target_globals:
            continue
        try:
            target_globals[name] = getattr(mod, name)
            exported.append(name)
        except Exception:
            # Skip attributes that raise on access
            continue

    return exported
