import logging
from typing import Dict, Optional

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
    """

    def __init__(self, base_dataset: CoreDataset, size: int):
        super().__init__()
        self.base_dataset = base_dataset
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int, context: Optional[Dict] = None) -> Optional[Batch]:
        base_index = index % len(self.base_dataset)
        return self.base_dataset.__getitem__(base_index, context)
