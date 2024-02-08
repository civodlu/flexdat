from collections import defaultdict
from typing import Dict, Optional

from flexdat.dataset import CoreDataset
from flexdat.types import Batch


class DatasetAccessCounter(CoreDataset):
    def __init__(self, base_dataset: CoreDataset) -> None:
        super().__init__()
        self.counter = defaultdict(lambda: 0)
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index, context: Optional[Dict] = None) -> Batch:
        self.counter[index] += 1
        return self.base_dataset[index]
