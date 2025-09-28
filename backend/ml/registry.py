"""Lightweight JSON file-based model registry.

Provides a tiny API to register model metadata into a JSON file. This is
intentionally simple so it can be used in CI and local dev without heavy deps.
"""
import json
import os
from typing import Dict, Any


def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def register_model(registry_path: str, model_entry: Dict[str, Any]):
    """Append a model entry to the registry JSON file (list of entries).

    If the file doesn't exist, it will be created.
    """
    _ensure_dir(registry_path)
    entries = []
    if os.path.exists(registry_path):
        try:
            with open(registry_path, "r", encoding="utf-8") as fh:
                entries = json.load(fh)
        except Exception:
            entries = []

    entries.append(model_entry)
    with open(registry_path, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2)

    return True
