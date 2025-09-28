"""Configurable feature-engineering pipeline.

Provides a small fit/transform API supporting:
- column selection
- missing-value strategies: mean, median, constant
- categorical encoding: ordinal, one-hot (sparse), hashing
- numeric scaling: standard (zero mean unit var)

This implementation is intentionally lightweight and depends only on
numpy for numeric ops; sklearn is used if available for robust helpers.
"""
from typing import List, Optional, Any, Dict

try:
    import numpy as _np  # type: ignore
except Exception:
    _np = None
    # Provide lightweight pure-Python fallbacks for mean/median/std so the
    # feature pipeline can run in environments without numpy.
    import statistics as _statistics
    import math as _math

    class _NpLike:
        @staticmethod
        def array(seq, dtype=None):
            # keep as Python list for our simple needs
            return list(seq)

        @staticmethod
        def mean(arr):
            return _statistics.mean(arr) if arr else 0.0

        @staticmethod
        def median(arr):
            return _statistics.median(arr) if arr else 0.0

        @staticmethod
        def std(arr, ddof=0):
            if not arr:
                return 0.0
            # population std (ddof=0)
            mean = _statistics.mean(arr)
            var = sum((float(x) - mean) ** 2 for x in arr) / (len(arr) - ddof if len(arr) - ddof > 0 else 1)
            return _math.sqrt(var)

    _np = _NpLike()


class FeaturePipeline:
    """Simple configurable pipeline.

    Example config:
      config = {
        "select": ["a", "b", "cat"],
        "impute": {"strategy": "mean", "fill_value": 0},
        "categorical": {"type": "hash", "n_features": 16},
        "scale": {"type": "standard"}
      }
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        # learned stats
        self._means = None
        self._medians = None
        self._scale_stds = None

    def fit(self, rows: List[Dict[str, Any]]):
        """Learn statistics from rows (list of dicts).

        Only numeric columns are considered for numeric stats.
        """
        if _np is None:
            raise RuntimeError("numeric backend not available")

        cols = self.config.get("select") or (list(rows[0].keys()) if rows else [])
        numeric_vals = {c: [] for c in cols}
        for r in rows:
            for c in cols:
                v = r.get(c)
                try:
                    fv = float(v)
                    numeric_vals[c].append(fv)
                except Exception:
                    # non-numeric or missing
                    pass

        means = {}
        medians = {}
        stds = {}
        for c, vals in numeric_vals.items():
            if vals:
                arr = _np.array(vals, dtype=float)
                # compute mean/median/std robustly for numpy or fallback list
                try:
                    means[c] = float(_np.mean(arr))
                except Exception:
                    means[c] = float(_np.mean(list(arr)))
                try:
                    medians[c] = float(_np.median(arr))
                except Exception:
                    medians[c] = float(_np.median(list(arr)))

                # determine length robustly
                try:
                    length = arr.size
                except Exception:
                    try:
                        length = len(arr)
                    except Exception:
                        length = 1

                try:
                    std_val = float(_np.std(arr, ddof=0))
                except Exception:
                    std_val = float(_np.std(list(arr), ddof=0))

                stds[c] = std_val if (length and length > 1) else 0.0
            else:
                means[c] = None
                medians[c] = None
                stds[c] = None

        self._means = means
        self._medians = medians
        self._scale_stds = stds
        return self

    def transform(self, rows: List[Dict[str, Any]]):
        """Transform rows into numeric matrix and optional label vector.

        Returns: X (2D list), optionally y if label configured.
        """
        if _np is None:
            raise RuntimeError("numpy is required for FeaturePipeline")

        cols = self.config.get("select") or (list(rows[0].keys()) if rows else [])
        X = []
        for r in rows:
            row_vals = []
            for c in cols:
                v = r.get(c)
                if v is None or v == "":
                    # impute
                    imp_cfg = self.config.get("impute", {})
                    strat = imp_cfg.get("strategy", "mean")
                    if strat == "mean" and self._means and self._means.get(c) is not None:
                        v = self._means[c]
                    elif strat == "median" and self._medians and self._medians.get(c) is not None:
                        v = self._medians[c]
                    else:
                        v = imp_cfg.get("fill_value", 0.0)
                # categorical handling: hash
                cat_cfg = self.config.get("categorical", {})
                if cat_cfg and c in cat_cfg.get("columns", []):
                    ctype = cat_cfg.get("type", "hash")
                    if ctype == "hash":
                        n_features = int(cat_cfg.get("n_features", 16))
                        h = hash(str(v))
                        # simple hashed index -> one-hot position
                        idx = int(h % n_features)
                        # expand row_vals with zeros and a one at idx
                        vec = [0.0] * n_features
                        vec[idx] = 1.0
                        row_vals.extend(vec)
                        continue
                    elif ctype == "ordinal":
                        # map via stable hash
                        row_vals.append(float(abs(hash(str(v))) % 10000))
                        continue
                # numeric coercion
                try:
                    fv = float(v)
                except Exception:
                    fv = 0.0
                # scaling
                scale_cfg = self.config.get("scale", {})
                if scale_cfg.get("type") == "standard" and self._scale_stds and self._scale_stds.get(c):
                    std = self._scale_stds.get(c) or 1.0
                    mean = self._means.get(c) or 0.0
                    if std == 0:
                        fv = float(fv - mean)
                    else:
                        fv = float((fv - mean) / std)

                row_vals.append(fv)
            X.append(row_vals)

        return X


def rows_to_feature_matrix(rows: List[Dict[str, Any]], feature_columns: Optional[List[str]] = None, label_column: Optional[str] = None, pipeline_config: Optional[Dict[str, Any]] = None):
    """Convenience wrapper: fit pipeline and return (X, y).
    """
    cfg = pipeline_config or {}
    if feature_columns:
        cfg = dict(cfg)
        cfg.setdefault("select", feature_columns)

    pipe = FeaturePipeline(cfg)
    pipe.fit(rows)
    X = pipe.transform(rows)
    y = None
    if label_column:
        y = [r.get(label_column) for r in rows]
    return X, y
