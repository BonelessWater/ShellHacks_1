#!/usr/bin/env python3
"""Run quick checks for the feature pipeline without pytest."""
import sys
import traceback

def main():
    try:
        # Import the module by file path to avoid executing package-level
        # backend/__init__.py which pulls many optional dependencies.
        import importlib.util
        import pathlib

        repo_root = pathlib.Path(__file__).resolve().parents[1]
        fp = repo_root / "backend" / "ml" / "feature_pipeline.py"
        spec = importlib.util.spec_from_file_location("feature_pipeline", str(fp))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        FeaturePipeline = mod.FeaturePipeline
        rows_to_feature_matrix = mod.rows_to_feature_matrix

        # Test 1
        rows = [
            {"a": 1, "b": 10},
            {"a": None, "b": 14},
            {"a": 3, "b": None},
        ]
        cfg = {"select": ["a", "b"], "impute": {"strategy": "mean"}, "scale": {"type": "standard"}}
        p = FeaturePipeline(cfg)
        p.fit(rows)
        X = p.transform(rows)
        assert len(X) == 3
        assert all(len(r) == 2 for r in X)

        # Test 2
        rows2 = [
            {"x": 1, "y": 0},
            {"x": 2, "y": 1},
        ]
        X2, y2 = rows_to_feature_matrix(rows2, feature_columns=["x"], label_column="y")
        assert len(X2) == 2
        assert y2 == [0, 1]

        print("OK")
        return 0
    except Exception as e:
        traceback.print_exc()
        print(f"ERROR: {e}")
        return 2

if __name__ == '__main__':
    sys.exit(main())
