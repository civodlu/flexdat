from typing import Dict, List, Optional, Sequence

from .dataset import CoreDataset
from .types import Batch


class DatasetDict(CoreDataset):
    """
    Dataset based on a sequence of dictionary

    Example:

    >>> dataset = DatasetDict([{'v0': 0, 'v1': 1}, {'v0': 10, 'v1': 11}])
    >>> assert len(dataset) == 2
    >>> assert dataset[1]['v1'] == 11

    Used as list:

    >>> dataset = DatasetDict()
    >>> dataset.append({'v0': 0, 'v1': 1})
    >>> assert len(dataset) == 1

    """

    def __init__(self, values: Optional[List[Optional[Dict]]] = None):
        super().__init__()

        self.values: List[Optional[Dict]] = values if values is not None else []
        assert hasattr(self.values, 'append'), 'Must support `append` method!'

    def __len__(self) -> int:
        return len(self.values)

    def append(self, value: Dict) -> None:
        self.values.append(value)

    def __getitem__(self, index: int, context: Optional[Dict] = None) -> Optional[Batch]:
        return self.values[index]
