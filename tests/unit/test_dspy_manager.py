import importlib.util
from pathlib import Path


def _load_manager():
    repo_root = Path(__file__).resolve().parents[2]
    mgr_path = repo_root / "backend" / "dspy_manager.py"
    spec = importlib.util.spec_from_file_location("backend.dspy_manager", str(mgr_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_manager()


def test_dspy_manager_no_deps():
    import os
    mgr = _load_manager()
    # Remove GOOGLE_API_KEY for deterministic behavior
    os.environ.pop("GOOGLE_API_KEY", None)
    res = mgr.initialize_modules()
    assert res in (True, False)
