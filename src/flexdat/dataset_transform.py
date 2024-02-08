import logging
from typing import Callable, Dict, Optional

from .dataset import CoreDataset
from .types import Batch

logger = logging.getLogger(__name__)


class DatasetTransform(CoreDataset):
    """
    Transform a dataset.

    Example:

    >>> paths = ['/path/1', '/path/2', '/path/3']
    >>> dataset = DatasetPath(paths)
    >>> dataset = DatasetTransform(dataset, transform=lambda batch: batch)
    """

    def __init__(self, base_dataset: CoreDataset, transform: Callable[[Batch], Batch]):
        super().__init__()
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int, context: Optional[Dict] = None) -> Optional[Batch]:
        batch = self.base_dataset.__getitem__(index, context)
        if batch is None:
            return None
        batch_tfm = self.transform(batch)
        return batch_tfm
