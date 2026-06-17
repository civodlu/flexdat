from copy import copy
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from flexdat.dataset import Batch, CoreDataset, NonDeterministicDataset
from flexdat.dataset_cached_h5 import Sampler


class DatasetCachedPool(NonDeterministicDataset):
    """
    This dataset caches entries from another dataset to perform costly operations (e.g., augmentation, registration)

    Each entries may be reused a fixed number of times before being refreshed from the base dataset.

    This is designed to be used with `DatasetCachedH5` or similar.
    """

    def __init__(
        self,
        dataset: CoreDataset,
        cache_size: int = 30,
        number_of_reuse_per_entry: int = 10,
        pre_transform: Optional[Callable[[Batch], Batch]] = None,
        transform: Optional[Callable[[Batch], Batch]] = None,
        sampler: Optional[Sampler] = None,
        default_context: Dict = {
            'dataset_h5_disable_sampler': True,
            'dataset_h5_disable_transform': True,
        },
    ):
        """
        Args:
            dataset: the base dataset to cache
            cache_size: the number of entries to cache
            number_of_reuse_per_entry: the number of times to reuse an entry before reloading
            pre_transform: a function to apply to the batch before caching
            transform: a function to apply to the batch after caching
            sampler: a function to apply to the batch after caching and transform
            default_context: the context to use when loading from the base dataset
        """
        super().__init__()
        self.dataset = dataset
        self.cache_size = cache_size
        self.number_of_reuse_per_entry = number_of_reuse_per_entry
        self.pre_transform = pre_transform
        self.transform = transform
        self.cache: List[Tuple[int, int, Batch]] = [(number_of_reuse_per_entry, None, {}) for i in range(cache_size)]
        self.sampler = sampler
        self.default_context = default_context

        if len(dataset) == 0:
            raise ValueError('Dataset is empty!')

    def __getitem__(self, index: int, context: Optional[Dict] = {}) -> Optional[Batch]:
        assert index < self.cache_size
        reuse_count, base_index, batch = self.cache[index]
        if reuse_count >= self.number_of_reuse_per_entry:
            # reload a new entry
            base_index = np.random.randint(0, len(self.dataset))
            batch = self.dataset.__getitem__(base_index, self.default_context)
            if batch is None:
                return None
            if self.pre_transform is not None:
                batch = self.pre_transform(batch)
            self.cache[index] = (1, base_index, copy(batch))
        else:
            # reuse the cached entry
            self.cache[index] = (reuse_count + 1, base_index, copy(batch))

        if self.sampler is not None:
            batch = self.sampler(batch, context)

        if self.transform is not None:
            batch = self.transform(batch)

        batch['dataset_cached_pool_index'] = base_index
        return batch

    def __len__(self) -> int:
        return self.cache_size
