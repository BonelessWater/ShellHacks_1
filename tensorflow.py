"""Minimal TensorFlow shim for unit tests.

Implements a small subset of TensorFlow used by tests:
- tf.data.Dataset.from_tensor_slices
- Dataset.batch, prefetch, take, map, cache
- tf.data.AUTOTUNE
- tf.io.read_file, tf.image.decode_jpeg, tf.image.resize
- tf.constant, tf.ones

This file intentionally keeps implementation minimal and pure-Python.
"""
from __future__ import annotations

import itertools
from typing import Any, Iterable, Iterator, Tuple, Callable

import numpy as np


class _Dataset:
    def __init__(self, data: Iterable):
        self._data = list(data)

    @staticmethod
    def from_tensor_slices(tensors):
        if isinstance(tensors, tuple):
            features, labels = tensors
            combined = list(zip(list(features), list(labels)))
        else:
            combined = list(tensors)
        return _Dataset(combined)

    def __iter__(self) -> Iterator:
        return iter(self._data)

    def batch(self, batch_size: int):
        def gen():
            it = iter(self._data)
            while True:
                batch = list(itertools.islice(it, batch_size))
                if not batch:
                    break
                # If each element is a (feature, label) pair, separate them
                if len(batch) > 0 and isinstance(batch[0], (list, tuple)) and len(batch[0]) == 2:
                    feats = [x[0] for x in batch]
                    labs = [x[1] for x in batch]
                    # Convert to numpy arrays for stable behavior in tests
                    yield (np.asarray(feats), np.asarray(labs))
                else:
                    yield batch

        return _IterableDataset(gen)

    def prefetch(self, _):
        return self

    def take(self, n: int):
        return _Dataset(self._data[:n])

    def map(self, func: Callable, num_parallel_calls=None):
        def gen():
            for item in self._data:
                if isinstance(item, (list, tuple)):
                    yield func(*item)
                else:
                    yield func(item)

        return _IterableDataset(gen)

    def cache(self):
        return self


class _IterableDataset(_Dataset):
    def __init__(self, gen_func):
        self._gen_func = gen_func

    def __iter__(self) -> Iterator:
        return self._gen_func()

    def take(self, n: int):
        # Build a list of up to n items from the generator
        collected = []
        it = self._gen_func()
        for _ in range(n):
            try:
                collected.append(next(it))
            except StopIteration:
                break
        return _Dataset(collected)


class data:
    Dataset = _Dataset
    AUTOTUNE = -1


class io:
    @staticmethod
    def read_file(path: str) -> bytes:
        return b""


class image:
    @staticmethod
    def decode_jpeg(data: bytes, channels: int = 3):
        return np.ones((224, 224, channels), dtype=np.uint8)

    @staticmethod
    def resize(image_array: np.ndarray, size: Tuple[int, int]):
        h, w = size
        ch = image_array.shape[2] if image_array.ndim == 3 else 3
        return np.ones((h, w, ch), dtype=np.uint8)


def constant(value: Any):
    return value


def ones(shape: Tuple[int, ...]):
    return np.ones(shape)


def cast(x, dtype):
    # simple cast wrapper to numpy dtype
    import numpy as _np
    if hasattr(x, 'astype'):
        # dtype might be like 'float32' or np.float32
        try:
            return x.astype(_np.dtype(dtype))
        except Exception:
            return x.astype(_np.float32)
    return _np.array(x, dtype=_np.dtype(dtype))


class float32:
    pass


# Bindings

io = io
image = image


class tf:
    data = data
    io = io
    image = image
    constant = staticmethod(constant)
    ones = staticmethod(ones)
    AUTOTUNE = -1
    float32 = float32


# Expose Dataset for tf.data.Dataset usage in tests
Dataset = _Dataset
