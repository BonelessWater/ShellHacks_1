"""Minimal Dataset and DataLoader implementations for tests.

This is intentionally simple: Dataset is a wrapper around a list of items and DataLoader
iterates over the dataset, optionally batching items.
"""
from typing import Iterable, Iterator, List, Sequence, Optional


class Dataset(Sequence):
    """Simple sequence-like dataset."""
    def __init__(self, data: Iterable):
        self._data = list(data)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


class DataLoader:
    """A minimal DataLoader that yields items or batches from a Dataset.

    Accepts extra kwargs like num_workers and shuffle but ignores them in this shim.
    """
    def __init__(self, dataset: Iterable, batch_size: Optional[int] = None, shuffle: bool = False, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self) -> Iterator:
        if self.batch_size is None:
            for item in self.dataset:
                yield item
        else:
            batch = []
            for item in self.dataset:
                batch.append(item)
                if len(batch) >= self.batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch

    def __len__(self) -> int:
        if self.batch_size is None:
            return len(self.dataset)
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
