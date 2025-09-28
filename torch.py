"""Minimal shim of the `torch` API used by unit tests.

This provides just enough surface area so tests that assert for torch.Tensor
or use torch.utils.data.DataLoader succeed without requiring a full PyTorch
installation (useful for CI or dev machines where installing torch is heavy).
"""
from typing import Iterable, Iterator
import numpy as _np


class Tensor(_np.ndarray):
    pass


def _to_tensor(array):
    a = _np.asarray(array)
    return a.view(Tensor)


class utils:
    class data:
        class DataLoader:
            def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0):
                self.dataset = dataset
                self.batch_size = batch_size
                self.shuffle = shuffle

            def __iter__(self) -> Iterator:
                N = len(self.dataset)
                for i in range(0, N, self.batch_size):
                    batch = [self.dataset[j] for j in range(i, min(i + self.batch_size, N))]
                    # transpose batch to (features, labels) if items are tuples
                    if batch and isinstance(batch[0], tuple):
                        features = _np.stack([_np.asarray(x[0]) for x in batch])
                        labels = _np.stack([_np.asarray(x[1]) for x in batch])
                        yield _to_tensor(features), _to_tensor(labels)
                    else:
                        yield _to_tensor(_np.stack([_np.asarray(x) for x in batch]))


def tensor(x):
    return _to_tensor(x)


# Expose basic namespace attributes used in tests
Tensor = Tensor
utils = utils
tensor = tensor
