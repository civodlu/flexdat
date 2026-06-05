from typing import Dict, Optional, Sequence

from .dataset import CoreDataset
from .types import Batch


class DatasetInjectMetadata(CoreDataset):
    """
    Dataset concatenate metadata from a metadata dictionary.

    Example:

    >>> dataset = DatasetDict([{'uid': 'id_1', 'value_2': 'v2_1'}, {'uid': 'id_0', 'value_2': 'v2_0'}])
    >>> metadata = {'id_0': {'value': 'v0'}, 'id_1': {'value': 'v1'}}
    >>> dataset = DatasetInjectMetadata(dataset, metadata, key='uid')
    >>> assert len(dataset) == 2
    >>> assert dataset[0]['value'] == 'v1'
    """

    def __init__(
        self,
        base_dataset: CoreDataset,
        metadata: Dict[str, Dict],
        key: str,
        raise_on_missing: bool = True,
    ):
        super().__init__()
        self.base_dataset = base_dataset
        self.metadata = metadata
        self.key = key
        self.raise_on_missing = raise_on_missing

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int, context: Optional[Dict] = None) -> Optional[Batch]:
        batch = self.base_dataset.__getitem__(index, context)
        if batch is None:
            return None

        # we MUST have the key in the batch to be able to inject the metadata
        uid = batch[self.key]

        metadata = self.metadata.get(uid)
        if metadata is None:
            if self.raise_on_missing:
                raise KeyError(f"Metadata for uid {uid} not found in metadata dictionary")
            else:
                return batch

        batch.update(metadata)
        return batch

    def get_base_datasets(self) -> Sequence['CoreDataset']:
        return (self.base_dataset,)
