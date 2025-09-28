"""Utilities to load trained models and scalers and produce prediction callables.

The loader is guarded so agents can import it without requiring TensorFlow
or other heavy libraries at import time.
"""
from typing import Callable, Any, Optional


def create_predictor(model_path: str = None, scaler_path: str = None) -> Callable[[list], list]:
    """Return a callable predictor(features_list) -> list(scores)

    If TensorFlow is available and model_path points to a saved TF model,
    it will be loaded. Otherwise, this returns a dummy predictor raising
    a helpful error when called.
    """
    scaler = None
    model = None

    # Lazy imports
    try:
        from backend.ml.model_utils import load_preprocessor
        scaler = load_preprocessor(scaler_path) if scaler_path else None
    except Exception:
        scaler = None

    try:
        import tensorflow as tf  # type: ignore
        if model_path:
            model = tf.keras.models.load_model(model_path)
    except Exception:
        model = None

    def predictor(features_list: list) -> list:
        # features_list: list[dict] or 2d-array
        X = None
        try:
            import numpy as _np  # type: ignore
            # Attempt to transform dicts to arrays
            if isinstance(features_list[0], dict):
                keys = list(features_list[0].keys())
                X = _np.array([[f.get(k, 0.0) for k in keys] for f in features_list], dtype=float)
            else:
                X = _np.array(features_list, dtype=float)
        except Exception:
            X = features_list

        if scaler and hasattr(scaler, "transform"):
            try:
                X = scaler.transform(X)
            except Exception:
                pass

        if model is not None and hasattr(model, "predict"):
            out = model.predict(X)
            # flatten
            try:
                import numpy as _np  # type: ignore
                return list(_np.array(out).reshape(-1))
            except Exception:
                return list(out)

        raise RuntimeError("No model available. Install TensorFlow and provide a model_path to create_predictor.")

    return predictor
