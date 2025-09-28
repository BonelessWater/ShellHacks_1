"""Minimal shim of the 'torch' package used for tests when real PyTorch isn't installed.

This provides a tiny subset: Tensor, tensor(), FloatTensor and a utils.data module with
Dataset and DataLoader classes that behave like simple Python iterables. It's intentionally
minimal and only exists to allow test collection in environments where installing PyTorch
is infeasible.
"""
from .core import Tensor, tensor, FloatTensor
from . import utils

__all__ = ["Tensor", "tensor", "FloatTensor", "utils"]
