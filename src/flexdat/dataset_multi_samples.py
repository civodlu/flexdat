import logging
from typing import Callable, Dict, Literal, Optional, Sequence, Union

import numpy as np

from .dataset import CoreDataset
from .types import Batch

logger = logging.getLogger(__name__)


def to_base(number: int, base: int, min_size: int) -> Sequence[int]:
    """Converts a non-negative number to a list of digits in the given base.

    The base must be an integer greater than or equal to 2 and the first digit
    in the list of digits is the most significant one.
    """
    if not number:
        return [0] * min_size

    digits = []
    while number:
        digits.append(number % base)
        number //= base
    digits = list(reversed(digits))
    padding = min_size - len(digits)
    return [0] * padding + digits


class DatasetMultiSample(CoreDataset):
    """
    Combine multiple samples into a single sample.

    This is deterministic sampling.

    Example:
    >>> from torch.utils.data.dataloader import default_collate
    >>> paths = ['/path/1', '/path/2', '/path/3']
    >>> dataset = DatasetPath(paths)
    >>> dataset = DatasetMultiSample(dataset, nb_samples=5, collate_fn=default_collate)
    >>> assert len(dataset) == 5 ** 3
    """

    def __init__(
        self,
        base_dataset: CoreDataset,
        nb_samples: int,
        collate_fn: Callable[[Sequence[Batch]], Batch],
    ):
        """
        Args:
            nb_samples: the number of samples to be
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.nb_samples = nb_samples
        self.collate_fn = collate_fn

    def __len__(self) -> int:
        return len(self.base_dataset) ** self.nb_samples

    def __getitem__(self, index: int, context: Optional[Dict] = None) -> Optional[Batch]:
        assert index < len(self)
        indices = to_base(index, len(self.base_dataset), min_size=self.nb_samples)
        batches = [self.base_dataset.__getitem__(i, context=context) for i in indices]
        return self.collate_fn(batches)
