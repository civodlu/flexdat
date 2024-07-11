import logging
from typing import Dict, Literal, Optional, Sequence, Union

import numpy as np

from .dataset import CoreDataset
from .types import Batch

logger = logging.getLogger(__name__)


class DatasetResizeVirtual(CoreDataset):
    """
    Make the dataset appear to be of a different size by repeating or discarding
    indices.

    Example:

    >>> paths = ['/path/1', '/path/2', '/path/3']
    >>> dataset = DatasetPath(paths)
    >>> dataset = DatasetResizeVirtual(dataset, size=10)
    >>> assert dataset[3]['path'] == '/path/1'

    Optionally, multiple datasets can be supported.

    >>> paths1 = ['/path/1', '/path/2', '/path/3']
    >>> dataset1 = DatasetPath(paths1)
    >>> paths2 = ['/path/1', '/path/2', '/path/3']
    >>> dataset2 = DatasetPath(paths2)
    >>> dataset = DatasetResizeVirtual([dataset1, dataset2], size=10)
    >>> assert dataset[0]['path'] in paths1
    >>> assert dataset[1]['path'] in paths2
    """

    def __init__(
        self,
        base_dataset: Union[CoreDataset, Sequence[CoreDataset]],
        size: int,
        sampling_mode: Literal['sequential'] = 'sequential',
    ):
        """
        Args:
            sampling_mode: `sequential` will sample from each base_datasets sequentially
        """
        super().__init__()

        if not isinstance(base_dataset, Sequence):
            base_dataset = [base_dataset]
        self.base_datasets = base_dataset
        self.size = size
        self.sampling_mode = sampling_mode

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int, context: Optional[Dict] = None) -> Optional[Batch]:
        assert index < self.size
        base_dataset_index = index % len(self.base_datasets)
        if self.sampling_mode == 'sequential':
            base_dataset = self.base_datasets[base_dataset_index]
            base_index = np.random.randint(0, len(base_dataset))
            return base_dataset.__getitem__(base_index, context)

        raise ValueError(f'unsupported sampling={self.sampling_mode}!')
