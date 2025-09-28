"""Minimal core utilities for the torch shim."""
from typing import Any, Iterable, Iterator, List, Optional
import numpy as _np


class Tensor:
    """A tiny stand-in for torch.Tensor. Stores data as Python objects.

    Provides a .numpy() method to return underlying numpy array or Python data.
    """
    def __init__(self, data: Any):
        self.data = data

    def numpy(self):
        return self.data

    def __repr__(self) -> str:  # pragma: no cover - simple repr
        return f"Tensor({self.data!r})"


def tensor(data: Any, dtype: Optional[Any] = None) -> Tensor:
    """Create a Tensor-like object from Python data.

    dtype: optional numpy dtype (e.g., _np.float32) to coerce the data.
    """
    if dtype is not None:
        try:
            arr = _np.asarray(data, dtype=dtype)
        except Exception:
            arr = _np.asarray(data)
    else:
        arr = _np.asarray(data)
    return Tensor(arr)


def FloatTensor(data: Any) -> Tensor:
    """Return a Tensor backed by a float32 numpy array.

    Accepts array-like input and returns a Tensor wrapping numpy.asarray(..., dtype=float32).
    """
    return tensor(data, dtype=_np.float32)
