import logging
from itertools import chain
from typing import Callable, Dict, Optional, Sequence

from .dataset import CoreDataset
from .sampler_pairing import Pairing, PairingSampler, PairingSamplerRandom
from .types import Batch

logger = logging.getLogger(__name__)


class DatasetPaired(CoreDataset):
    """
    Group samples of a dataset in a paired dataset.

    The new dataset will have a length EQUAL to the number of pairings.

    For example:
    >>> dataset = DatasetPath(['location_1_full', 'location_2_full', 'location_1_low', 'location_2_low'])
    >>> pairing_sampler = RandomPairingSampler(pairing_key_prefix=('full_', 'low_'))
    >>> dataset = DatasetPaired(dataset, [(0, 2), (1, 3)], pairing_sampler=pairing_sampler)
    """

    def __init__(
        self,
        base_dataset: CoreDataset,
        pairing: Pairing,
        pairing_sampler: PairingSampler = PairingSamplerRandom(),
        keys_to_keep: Optional[Sequence[str]] = None,
        keys_to_discard: Optional[Sequence[str]] = None,
        pairing_metadata: Optional[Dict] = None,
        transform: Optional[Callable[[Batch], Batch]] = None,
    ) -> None:
        """
        Args:
            pairing: pairing of indices. This redefine the size of the dataset. Can be pairs, triplets, etc...
            pairing_sampler: specify how to sample the pairing
            keys_to_keep: the only keys to keep for the resulting batch
            keys_to_discard: the keys to be removed from the resulting batches
            pairing_metadata: metadata to be added to the batch
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.pairing = pairing
        self.pairing_sampler = pairing_sampler
        self.keys_to_keep = keys_to_keep
        self.keys_to_discard = keys_to_discard
        if self.keys_to_discard is not None:
            assert not isinstance(self.keys_to_discard, str), 'should be a sequence of string!!!!'

        self.transform = transform
        self.pairing_metadata = pairing_metadata

        max_pairing_id = max(chain(*pairing))
        assert max_pairing_id < len(
            self.base_dataset
        ), f'index={max_pairing_id} is not present in the base dataset. Max allowed index={len(self.base_dataset)}'

        if pairing_metadata is not None:
            for name, value in pairing_metadata.items():
                assert len(value) == len(pairing)

    def __len__(self) -> int:
        return len(self.pairing)

    def __getitem__(self, index: int, context: Optional[Dict] = None) -> Optional[Batch]:
        final_batch = {}

        if context is None:
            # here create a context: most likely we need to sync
            # samplers coordinates
            context = {}

        pairings = self.pairing_sampler(
            base_index=index,
            pairing=self.pairing,
            context=context,
            pairing_metadata=self.pairing_metadata,
        )

        for key_prefix, i in pairings:
            batch = self.base_dataset.__getitem__(i, context)
            if batch is None:
                return None

            if self.keys_to_keep:
                keys = self.keys_to_keep
            else:
                keys = batch.keys()  # type: ignore

            for k in keys:
                if self.keys_to_discard and k in self.keys_to_discard:
                    continue
                final_batch[key_prefix + k] = batch[k]

        if self.pairing_metadata is not None:
            for key, value in self.pairing_metadata.items():
                final_batch[key] = value[index]

        if self.transform is not None:
            final_batch = self.transform(final_batch)

        return final_batch
