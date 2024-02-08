import collections
import logging
from itertools import chain
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .dataset import CoreDataset
from .dataset_pairing import Pairing
from .types import Batch

logger = logging.getLogger(__name__)


class DatasetPairingPreprocessor(CoreDataset):
    """
    Inject some metadata based on data pairing.

    For example, we often need to have a voxel to voxel mapping in multi-modal imaging (e.g., CT/MRI).
    Unfortunately, DICOM geometry may not be identical between these 2 images. In this case, based on the pairing,
    the target geometry for the resampling will be injected in the Batch. This can be used later by a `DatasetResample`.

    This dataset is designed to be jointly used with `DatasetPaired` / `DatasetPairedList` class.

    Often, this data will re-use multiple times the same index so they should be cached (e.g., `DatasetCachedMultiSamples`)
    to avoid duplicated computations.

    BEWARE, if the same samples are re-used between multiple pairings, the metadata extracted must be consistent!

    >>> dataset = DatasetPath(['location_1_full', 'location_2_full', 'location_1_low', 'location_2_low'])
    >>> def metadata_fn(batches):
            return {'data': [b['path'] for b in batches]}
    >>> associations = [(0, 2), (1, 3)]
    >>> dataset = DatasetPairingMetadataInjector(dataset, associations, metadata_fn)
    >>> dataset = DatasetPaired(
            dataset,
            associations,
            pairing_sampler=RandomPairingSampler(pairing_key_prefix=('full_', 'low_')))
    """

    def __init__(self, base_dataset: CoreDataset, pairing: Pairing, metadata_fn: Callable[[Sequence[Batch]], Batch]) -> None:
        super().__init__()
        self.base_dataset = base_dataset
        self.pairing = pairing
        self.metadata_fn = metadata_fn

        max_pairing_id = max(chain(*pairing))
        assert max_pairing_id < len(
            self.base_dataset
        ), f'index={max_pairing_id} is not present in the base dataset. Max allowed index={len(self.base_dataset)}'

        # build inverse mapping base_dataset index -> pairing index
        # if multiple pairs are involved with the same base_dataset index, those metadata MUST be consistent
        self.index_to_pairing = [None] * len(base_dataset)
        for indices_n, indices in enumerate(pairing):
            for i in indices:
                self.index_to_pairing[i] = indices_n  # type: ignore

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int, context: Optional[Dict] = None) -> Batch:
        if context is None:
            # here create a context: most likely we need to sync
            # samplers coordinates
            context = {}

        # find the pairing in the list of pairing
        pairing_index = self.index_to_pairing[index]
        if pairing_index is None:
            raise RuntimeError(
                f'`pairing_index` is None for index={index}. This is unexpected'
                ' as the pairing in the constructor did not mark it!'
            )
        paired_batches = [self.base_dataset.__getitem__(i, context) for i in self.pairing[pairing_index]]
        metadata = self.metadata_fn(paired_batches)

        batch = self.base_dataset.__getitem__(index, context)
        if batch is None:
            return None

        return {**batch, **metadata}
