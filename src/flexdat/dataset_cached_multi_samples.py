from copy import copy
from typing import Dict, List, Optional

from .dataset import CoreDataset
from .types import Batch


class DatasetCachedMultiSamples(CoreDataset):
    """
    Cache the last indices used.

    For example, this can be useful when pairing the same sample which is reconstructed
    with different algorithms. Here we want to cache the k-space, which is the same input
    for the pairing.

    Example:

    >>> dataset = DatasetDict([{'v0': 0, 'v1': 1}, {'v0': 10, 'v1': 11}], cache_size=3)
    >>> dataset = DatasetCachedMultiSamples(dataset)
    >>> assert len(dataset) == 2
    >>> assert dataset[1]['v1'] == 11
    >>> assert dataset[1]['v1'] == 11  # reused
    """

    def __init__(self, base_dataset: CoreDataset, cache_size: int = 10):
        super().__init__()
        self.base_dataset = base_dataset
        self.last_indices: List[int] = []
        self.last_samples: List[Optional[Batch]] = []
        self.cache_size = cache_size

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int, context: Optional[Dict] = None) -> Optional[Batch]:
        assert index is not None

        try:
            index_index = self.last_indices.index(index)
            return copy(self.last_samples[index_index])
        except ValueError:
            # not inside the cached
            pass

        # recalculate the cache
        sample = self.base_dataset.__getitem__(index, context)
        self.last_indices.append(index)
        self.last_samples.append(sample)
        if len(self.last_indices) > self.cache_size:
            # remove the oldest index
            self.last_indices.pop(0)
            self.last_samples.pop(0)

        # make a copy to make sure the original sample is NOT modified
        # as we will be re-using this sample multiple time
        return copy(sample)
