from typing import Dict, Optional, Sequence

from .dataset import CoreDataset
from .types import Batch


class DatasetMerge(CoreDataset):
    """
    Concatenate the keys of multiple datasets.

    For example, if we need to pre-calculate segmentations and add it
    to our existing dataset.

    Example:

    >>> dataset_1 = DatasetDict([{'v0': 0, 'v1': 1}, {'v0': 10, 'v1': 11}])
    >>> dataset_2 = DatasetDict([{'v2': 100, 'v3': 101}, {'v2': 110, 'v3': 111}])
    >>> dataset = DatasetMerge([dataset_1, dataset_2])
    >>> assert len(dataset) == 2
    >>> assert dataset[1]['v2'] == 110

    Example:

    >>> dataset_1 = DatasetDict([{'v0': 0, 'v1': 1}, {'v0': 10, 'v1': 11}])
    >>> dataset_2 = DatasetDict([{'v2': 100, 'v3': 101}, {'v2': 110, 'v3': 111}])
    >>> dataset = DatasetMerge([dataset_1, dataset_2], key_prefix=('dat0_', 'dat1_'))
    >>> assert len(dataset) == 2
    >>> assert dataset[1]['dat1_v2'] == 110

    """

    def __init__(
        self, datasets: Sequence[CoreDataset], key_prefix: Optional[Sequence[str]] = None, allow_key_collisions: bool = False
    ):
        """
        Parameters:
            allow_key_collisions: if True, colliding key will not raise exception
        """
        super().__init__()
        self.datasets = datasets
        self.key_prefix = key_prefix
        self.allow_key_collisions = allow_key_collisions
        if key_prefix:
            assert len(key_prefix) == len(datasets)

        for d in self.datasets[1:]:
            assert len(d) == len(
                datasets[0]
            ), f'merged dataset must have the exactly the same size! Got={len(d)}, expected={len(datasets[0])}'

    def __len__(self) -> int:
        return len(self.datasets[0])

    def __getitem__(self, index: int, context: Optional[Dict] = None) -> Optional[Batch]:
        batch: Batch = {}
        for dataset_n, dataset in enumerate(self.datasets):
            if self.key_prefix:
                prefix = self.key_prefix[dataset_n]
            else:
                prefix = ''

            dataset_item = dataset.__getitem__(index, context)
            if dataset_item is None:
                # if any dataset failed, the whole merge failed
                return None

            for key, values in dataset_item.items():
                full_key = prefix + key

                if not self.allow_key_collisions:
                    assert full_key not in batch, f'key collision={full_key}. Got={batch.keys()}'

                if full_key not in batch:
                    # keep only the first collision if any
                    batch[full_key] = values
        return batch
