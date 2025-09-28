"""Feature engineering helpers for training pipelines.

These helpers are intentionally small and dependency-free so they can run in
CI without heavy ML libraries. They provide a simple mapping from a list of
dict rows to numeric feature arrays and labels.
"""
from typing import List, Dict, Any, Tuple, Optional


def rows_to_feature_matrix(rows: List[Dict[str, Any]], feature_columns: Optional[List[str]] = None, label_column: Optional[str] = None) -> Tuple[List[List[float]], List[Any]]:
    """Convert list-of-dicts rows into X (2D numeric list) and y (1D list).

    - feature_columns: list of keys to use as features; if None uses all except label_column
    - label_column: key to use as label; if None, returns y as empty list

    This function coerces values to float when possible and fills missing with 0.0.
    Categorical strings are hashed deterministically to floats (stable across runs).
    """
    if not rows:
        return [], []

    # Determine columns
    all_keys = list(rows[0].keys())
    if label_column and label_column in all_keys:
        keys = [k for k in all_keys if k != label_column]
    else:
        keys = all_keys

    if feature_columns:
        keys = [k for k in feature_columns if k in all_keys]

    X = []
    y = []
    for r in rows:
        row_feat = []
        for k in keys:
            v = r.get(k)
            if v is None:
                row_feat.append(0.0)
                continue
            # Try numeric coercion
            try:
                row_feat.append(float(v))
                continue
            except Exception:
                pass

            # Fallback: hash strings deterministically
            try:
                s = str(v)
                h = 0
                for ch in s:
                    h = (h * 31 + ord(ch)) & 0xFFFFFFFF
                row_feat.append(float(h % 100000) / 100000.0)
            except Exception:
                row_feat.append(0.0)

        X.append(row_feat)
        if label_column:
            y.append(r.get(label_column))

    return X, y
