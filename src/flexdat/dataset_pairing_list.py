import collections
import logging
from itertools import chain
from typing import Any, Callable, Dict, Optional, Sequence

from .dataset import CoreDataset
from .dataset_pairing import Pairing, PairingSampler, PairingSamplerRandom
from .types import Batch

logger = logging.getLogger(__name__)


class DatasetPairedList(CoreDataset):
    """
    Group samples of a dataset into a single dataset. Keys will be pooled in a list inside the batch.

    For example:
    >>> dataset = DatasetPath(['location_1_full', 'location_2_full', 'location_1_low', 'location_2_low'])
    >>> pairing_sampler = RandomPairingSampler()
    >>> dataset = DatasetPairedList(dataset, [(0, 2), (1, 3)], pairing_sampler=pairing_sampler)
    >>> samples = dataset[0]
    >>> assert samples['path'][0] == 'location_1_full'
    >>> assert samples['path'][1] == 'location_1_low'
    """

    def __init__(
        self,
        base_dataset: CoreDataset,
        pairing: Pairing,
        pairing_sampler: PairingSampler = PairingSamplerRandom(),
        keys_to_keep: Optional[Sequence[str]] = None,
        keys_to_discard: Optional[Sequence[str]] = None,
        transform: Optional[Callable[[Batch], Any]] = None,
    ) -> None:
        """
        Args:
            pairing: pairing of indices. This redefine the size of the dataset. Can be paires, triplets, etc...
            pairing_sampler: specify how to sample the pairing
            keys_to_keep: the only keys to keep for the resulting batch
            keys_to_discard: the keys to be removed from the resulting batches
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.pairing = pairing
        self.pairing_sampler = pairing_sampler
        self.keys_to_keep = keys_to_keep
        self.keys_to_discard = keys_to_discard
        self.transform = transform

        max_pairing_id = max(chain(*pairing))
        assert max_pairing_id < len(
            self.base_dataset
        ), f'index={max_pairing_id} is not present in the base dataset. Max allowed index={len(self.base_dataset)}'

    def __len__(self) -> int:
        return len(self.pairing)

    def __getitem__(self, index: int, context: Optional[Dict] = None) -> Optional[Batch]:
        if context is None:
            # here create a context: most likely we need to sync
            # samplers coordinates
            context = {}

        pairings = self.pairing_sampler(
            base_index=index,
            pairing=self.pairing,
            context=context,
            pairing_metadata=None,
        )

        final_batch = collections.defaultdict(list)
        for _, index in pairings:
            batch = self.base_dataset.__getitem__(index, context)
            if batch is None:
                return None

            if self.keys_to_keep:
                keys = self.keys_to_keep
            else:
                keys = batch.keys()  # type: ignore

            for k in keys:
                if self.keys_to_discard and k in self.keys_to_discard:
                    continue
                final_batch[k].append(batch[k])

        if self.transform is not None:
            final_batch = self.transform(final_batch)

        return final_batch
