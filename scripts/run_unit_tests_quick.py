#!/usr/bin/env python3
"""Quick runner to execute a small set of unit tests without pytest."""
import importlib
import sys
import pathlib


def run_test(module_name, func_name):
    # Load test module by file path to avoid package import side-effects
    import importlib.util
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    module_path = repo_root / module_name.replace(".", "/")
    module_file = module_path.with_suffix(".py")
    spec = importlib.util.spec_from_file_location(module_name, str(module_file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    f = getattr(m, func_name)
    try:
        # Provide a tmp_path if function expects pytest fixture by using tempfile
        import tempfile
        from pathlib import Path

        tmp = Path(tempfile.mkdtemp())
        # Some tests accept tmp_path argument; try calling with and without
        try:
            f(tmp)
        except TypeError:
            f()
    except Exception as e:
        print(f"FAIL: {module_name}.{func_name} -> {e}")
        raise


def main():
    tests = [
        ("tests.unit.test_registry", "test_register_model_tmp"),
        ("tests.unit.test_dspy_manager", "test_dspy_manager_no_deps"),
    ]
    for mod, fn in tests:
        print(f"Running {mod}.{fn}...")
        run_test(mod, fn)
    print("All quick tests passed")


if __name__ == '__main__':
    # Ensure project root in path
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    main()
