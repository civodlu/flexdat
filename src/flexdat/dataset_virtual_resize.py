import logging
from typing import Dict, Literal, Optional, Sequence, Union

import numpy as np

from .dataset import CoreDataset, NonDeterministicDataset
from .types import Batch

logger = logging.getLogger(__name__)


class DatasetResizeVirtual(NonDeterministicDataset):
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
        retry_on_none: bool = True,
    ):
        """
        Args:
            sampling_mode: `sequential` will sample from each base_datasets sequentially
            retry_on_none: if True, the batch with None will be re-tried until a non-None batch is
                obtained using the same base dataset index. May still return None if the max retry is reached.
        """
        super().__init__()

        if not isinstance(base_dataset, Sequence):
            base_dataset = [base_dataset]
        self.base_datasets = base_dataset
        self.size = size
        self.sampling_mode = sampling_mode
        self.retry_on_none = retry_on_none
        self.max_retry = 10000

        for n, dataset in enumerate(self.base_datasets):
            if len(dataset) == 0:
                logger.warning(f'Base dataset {n} is empty! This may be unexpected!')

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int, context: Optional[Dict] = None) -> Optional[Batch]:
        assert index < self.size
        base_dataset_index = index % len(self.base_datasets)
        if self.sampling_mode == 'sequential':
            base_dataset = self.base_datasets[base_dataset_index]
            # here we retry until we get a batch. This dataset is often
            # used as a wrapper to sample from different datasets (e.g., stratification).
            # Ensure we get a sample from EACH base dataset
            for _ in range(self.max_retry):
                if len(base_dataset) == 0:
                    return None

                base_index = np.random.randint(0, len(base_dataset))
                b = base_dataset.__getitem__(base_index, context)
                if b is not None or self.retry_on_none is False:
                    return b

            logger.warning(f'Failed to get a non-None batch after {self.max_retry} retries for dataset index {index}!')
            return None

        raise ValueError(f'unsupported sampling={self.sampling_mode}!')

    def get_base_datasets(self) -> Sequence[CoreDataset]:
        return self.base_datasets
