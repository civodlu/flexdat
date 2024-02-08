import logging
from typing import Any, Dict, Optional, Sequence

from .dataset import CoreDataset
from .types import Batch

logger = logging.getLogger(__name__)


class DatasetPath(CoreDataset):
    """
    Dataset pointing to path to local drive. Optionally, record metadata.

    Example:

    >>> paths = ['/path/1', '/path/2']
    >>> dataset = DatasetPath(paths)

    Example:

    >>> paths = ['/path/1', '/path/2']
    >>> doses = [1.0, 0.1]
    >>> dataset = DatasetPath(paths, doses=doses)
    """

    def __init__(self, locations: Sequence[Optional[str]], **metadata: Any):
        super().__init__()
        self.locations = locations
        for key, values in metadata.items():
            assert len(values) == len(locations), (
                f'size mismatch. Metadata values must be exactly the same size as path.'
                f' Got={len(values)}, expected={len(locations)}'
            )
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.locations)

    def __getitem__(self, index: int, context: Optional[Dict] = None) -> Optional[Batch]:
        batch = {'path': self.locations[index]}
        for key, values in self.metadata.items():
            batch[key] = values[index]
        return batch
