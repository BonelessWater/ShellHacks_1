import os
import json
import importlib.util
from pathlib import Path


def _load_register_model():
    repo_root = Path(__file__).resolve().parents[2]
    reg_path = repo_root / "backend" / "ml" / "registry.py"
    spec = importlib.util.spec_from_file_location("backend.ml.registry", str(reg_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.register_model


def test_register_model_tmp(tmp_path):
    register_model = _load_register_model()
    registry_file = tmp_path / "registry.json"
    entry = {"model": "test", "metrics": {"acc": 0.9}}
    assert register_model(str(registry_file), entry) is True
    # Ensure file exists and contains the entry
    with open(registry_file, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    assert isinstance(data, list)
    assert data[0]["model"] == "test"
