import logging
from typing import Callable, Dict, Optional

import numpy as np

from .dataset import CoreDataset
from .types import Batch

logger = logging.getLogger(__name__)


class DatasetConcatenate(CoreDataset):
    """
    Concatenate multiple datasets to form a single dataset (i.e., resulting in a dataset with more samples)

    Example:

    >>> d0 = DatasetPath(['L0', 'L1'])
    >>> d1 = DatasetPath(['L2', 'L3', 'L4])
    >>> dataset = DatasetConcatenate({'sub_dataset0': d0, 'sub_dataset1': d1})
    >>> assert len(dataset) == 5
    """

    def __init__(
        self,
        datasets: Dict[str, CoreDataset],
        transform: Optional[Callable[[Batch], Batch]] = None,
        append_dataset_name: bool = True,
        dataset_name_key: str = 'dataset_name',
    ) -> None:
        super().__init__()
        self.dataset_names = list(datasets.keys())
        self.datasets = list(datasets.values())
        self.transform = transform
        self.append_dataset_name = append_dataset_name
        self.dataset_name_key = dataset_name_key

        size = 0
        bins = []
        for d in self.datasets:
            size += len(d)
            bins.append(size)

        self.size = size
        self.bins = bins

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int, context: Optional[Dict] = None) -> Optional[Batch]:
        dataset_index = np.digitize(index, self.bins)

        base_index = 0 if dataset_index == 0 else self.bins[dataset_index - 1]
        index_within_dataset: int = index - base_index
        batch = self.datasets[dataset_index].__getitem__(index_within_dataset, context)
        if batch is None:
            return None
        if self.append_dataset_name:
            batch[self.dataset_name_key] = self.dataset_names[dataset_index]

        if self.transform is not None:
            batch = self.transform(batch)
        return batch
