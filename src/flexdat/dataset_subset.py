import logging
from typing import Dict, Optional, Sequence

from .dataset import CoreDataset
from .types import Batch

logger = logging.getLogger(__name__)


class DatasetSubset(CoreDataset):
    """
    Restrict the dataset to a given subset.

    Example:

    >>> paths = ['/path/1', '/path/2', '/path/3']
    >>> dataset = DatasetPath(paths)
    >>> dataset = DatasetSubset(dataset, indices=[1, 2])
    >>> assert dataset[0]['path'] == '/path/2'
    """

    def __init__(self, base_dataset: CoreDataset, indices: Sequence[int]):
        super().__init__()
        self.base_dataset = base_dataset
        self.indices = indices
        assert max(indices) < len(
            base_dataset
        ), f'max_index={max(indices)} larger than base dataset! base={len(base_dataset)}'

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int, context: Optional[Dict] = None) -> Optional[Batch]:
        base_index = self.indices[index]
        batch = self.base_dataset.__getitem__(base_index, context)
        return batch
