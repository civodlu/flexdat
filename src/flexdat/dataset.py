from typing import Dict, Optional, Protocol, Sequence, runtime_checkable

from .types import Batch


@runtime_checkable
class CoreDataset(Protocol):
    """
    Base dataset class. This mimic the `torch.utils.data.Dataset` API.
    """

    def __getitem__(self, index: int, context: Optional[Dict] = None) -> Optional[Batch]:
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()

    def get_base_datasets(self) -> Sequence['CoreDataset']:
        """
        If this dataset is a wrapper around another dataset, return the base dataset.
        """
        return ()


class NonDeterministicDataset(CoreDataset):
    """
    A dataset that is non-deterministic, i.e. it can return different batches for the same index.

    This is useful for datasets that perform random sampling or data augmentation.

    Beware of using these datasets with caching or prefetching, as it can lead to unexpected behavior.
    """
