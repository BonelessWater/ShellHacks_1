"""Compatibility helpers for re-exporting archived backend modules.

This module centralizes the import-fallback pattern used to keep
`backend.<module>` imports working when implementations live under
`backend.archive.<module>` after a merge. Each shim file calls
`reexport_module("module_name")` to copy public symbols into the shim's
globals.
"""
from importlib import import_module
from types import ModuleType
from typing import List


def _load_archive_module(module_name: str) -> ModuleType:
    """Try to import the implementation from backend.archive.<module_name>."""
    return import_module(f"backend.archive.{module_name}")


def _load_top_level_backend(module_name: str) -> ModuleType:
    """Try to import the implementation from backend.<module_name> (legacy)."""
    return import_module(f"backend.{module_name}")


def reexport_module(module_name: str, target_globals: dict) -> List[str]:
    """Import a module from the archive (preferred) or fallback to
    backend.<module> and copy its public symbols into `target_globals`.

    Returns the list of exported names.
    """
    mod = None
    try:
        mod = _load_archive_module(module_name)
    except Exception:
        try:
            mod = _load_top_level_backend(module_name)
        except Exception:
            # Last resort: try importing a top-level sibling module (non-package)
            mod = import_module(module_name)

    exported = []
    for name in dir(mod):
        if name.startswith("_"):
            continue
        target_globals[name] = getattr(mod, name)
        exported.append(name)

    return exported
