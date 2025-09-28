"""Write a simple JSON manifest for model artifacts.

This records model_path, scaler_path, metadata_path, dataset fingerprint and other
useful provenance information.
"""
import json
import os
from typing import Dict, Any


def write_manifest(path: str, manifest: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
