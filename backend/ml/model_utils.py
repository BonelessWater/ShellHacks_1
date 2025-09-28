"""Simple preprocessing utilities for ML workflows.

These helpers are lightweight and guarded so tests can run without
sklearn/numpy installed. They provide a minimal StandardScaler-like
interface and helpers to save/load preprocessors.
"""
from typing import Any, Optional
import pickle
import os


class SimpleScaler:
    """A tiny scaler that computes mean/std and applies (x - mean)/std.

    Falls back to lists if numpy isn't available.
    """

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        try:
            import numpy as _np  # type: ignore

            arr = _np.array(X, dtype=float)
            self.mean_ = arr.mean(axis=0)
            self.scale_ = arr.std(axis=0)
            # avoid zero
            self.scale_[self.scale_ == 0] = 1.0
        except Exception:
            # pure python fallback
            cols = list(zip(*X))
            self.mean_ = [sum(c) / len(c) for c in cols]
            self.scale_ = []
            for c, m in zip(cols, self.mean_):
                s = (sum((v - m) ** 2 for v in c) / len(c)) ** 0.5 if len(c) else 1.0
                self.scale_.append(s or 1.0)

    def transform(self, X):
        try:
            import numpy as _np  # type: ignore

            arr = _np.array(X, dtype=float)
            return (arr - self.mean_) / self.scale_
        except Exception:
            out = []
            for row in X:
                out.append([(v - m) / s for v, m, s in zip(row, self.mean_, self.scale_)])
            return out

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def save_preprocessor(preprocessor: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(preprocessor, f)


def load_preprocessor(path: str) -> Optional[Any]:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)
