import os
from typing import Callable, Dict, Optional, Sequence

import h5py

from .dataset import CoreDataset
from .types import Batch


def bytes_to_string_(batch: Batch) -> Batch:
    for name, value in batch.items():
        if isinstance(value, bytes):
            batch[name] = value.decode('utf8')
    return batch


class DatasetReadH5(CoreDataset):
    """
    Read a H5 file and import all (optionally) specified keys

    Here we assume to have H5 created for us to use
    """

    def __init__(
        self,
        base_dataset: CoreDataset,
        keys: Optional[Sequence[str]] = None,
        path_name: str = 'path',
        transform: Optional[Callable[[Batch], Batch]] = bytes_to_string_,
    ):
        super().__init__()
        self.base_dataset = base_dataset
        self.keys = keys
        self.path_name = path_name
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int, context: Optional[Dict] = None) -> Optional[Batch]:
        batch = self.base_dataset.__getitem__(index, context)
        if batch is None:
            return None

        path = batch.get(self.path_name)

        assert path is not None, f'missing dataset key={self.path_name}'
        assert isinstance(path, str)
        assert os.path.exists(path), f'path={path} does not exist!'

        with h5py.File(path, 'r') as f:
            keys = self.keys
            if keys is None:
                keys = f.keys()

            for key in keys:
                value = f[key][()]
                batch[key] = value

        if self.transform is not None:
            batch = self.transform(batch)

        return batch
