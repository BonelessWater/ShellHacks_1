"""Cross-validation utilities (guarded).

Provides k-fold and stratified splitting helpers. Falls back to simple
implementations if sklearn is unavailable.
"""
from typing import Any, Iterable, List, Tuple


def k_fold_split(X: Iterable, y: Iterable, n_splits: int = 5, shuffle: bool = True, random_state: int = None):
    """Yield (train_idx, test_idx) pairs for k-fold CV.

    If sklearn is available, uses sklearn.model_selection.KFold otherwise
    falls back to a simple numpy/random implementation.
    """
    try:
        from sklearn.model_selection import KFold  # type: ignore
        import numpy as _np  # type: ignore

        X_arr = _np.arange(len(list(X)))
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        for train_idx, test_idx in kf.split(X_arr):
            yield train_idx.tolist(), test_idx.tolist()
        return
    except Exception:
        # simple fallback
        import random

        idx = list(range(len(list(X))))
        if shuffle:
            rnd = random.Random(random_state)
            rnd.shuffle(idx)
        fold_size = max(1, len(idx) // n_splits)
        for i in range(n_splits):
            start = i * fold_size
            end = start + fold_size if i < n_splits - 1 else len(idx)
            test_idx = idx[start:end]
            train_idx = [j for j in idx if j not in test_idx]
            yield train_idx, test_idx
