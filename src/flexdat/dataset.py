from typing import Dict, Optional, Protocol

from .types import Batch


class CoreDataset(Protocol):
    """
    Base dataset class. This mimic the `torch.utils.data.Dataset` API.
    """

    def __getitem__(self, index: int, context: Optional[Dict] = None) -> Optional[Batch]:
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()
