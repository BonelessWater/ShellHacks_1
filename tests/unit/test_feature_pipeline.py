from backend.ml.feature_pipeline import FeaturePipeline, rows_to_feature_matrix


def test_simple_numeric_impute_and_scale():
    rows = [
        {"a": 1, "b": 10},
        {"a": None, "b": 14},
        {"a": 3, "b": None},
    ]
    cfg = {"select": ["a", "b"], "impute": {"strategy": "mean"}, "scale": {"type": "standard"}}
    p = FeaturePipeline(cfg)
    p.fit(rows)
    X = p.transform(rows)
    # Expect 3 rows and 2 numeric columns
    assert len(X) == 3
    assert all(len(r) == 2 for r in X)


def test_rows_to_feature_matrix_returns_y():
    rows = [
        {"x": 1, "y": 0},
        {"x": 2, "y": 1},
    ]
    X, y = rows_to_feature_matrix(rows, feature_columns=["x"], label_column="y")
    assert len(X) == 2
    assert y == [0, 1]
