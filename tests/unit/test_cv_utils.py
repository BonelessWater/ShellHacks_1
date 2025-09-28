from backend.ml.cv_utils import k_fold_split


def test_k_fold_simple():
    X = list(range(10))
    y = [0 if i < 5 else 1 for i in X]
    folds = list(k_fold_split(X, y, n_splits=5, shuffle=False))
    # Expect 5 folds
    assert len(folds) == 5
    # Each test fold should cover roughly 2 items
    sizes = [len(test) for _, test in folds]
    assert sum(sizes) == 10

