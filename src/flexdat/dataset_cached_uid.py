import logging
import os
import time
from typing import Callable, Dict, Literal, Optional, Sequence

import h5py
from mypy_extensions import NamedArg

from .dataset import CoreDataset
from .types import Batch
from .utils_h5 import read_data_h5, write_data_h5

logger = logging.getLogger(__name__)


Sampler = Callable[[h5py.File], Batch]


def safe_uid(s: str) -> str:
    return (
        s.replace('/', '_')
        .replace('\\', '_')
        .replace('*', '_')
        .replace('$', '_')
        .replace('{', '_')
        .replace('}', '_')
        .replace('(', '_')
        .replace(')', '_')
    )


class DatasetCachedUID(CoreDataset):
    """
    Cache very long and STABLE computations (e.g., alignment between volumes or whole body segmentation
    provided by third party) using UIDs (rather than dataset index)

    Two datasets are collaborating:
    * the base_dataset holds the UIDs. If the cache for the corresponding UIDs are found, the cache is used
    * if the cache cannot be found or is out of version, `dataset_to_cache` is used to calculate what needs to be
      cached for the given UIDs

    For example:

    >>> dataset_uids = DatasetDict([{'uid': 'uid1'}, {'uid': 'uid2'}, {'uid': 'uid1'}])
    >>> dataset_to_cache = DatasetDict([{'key0': 'cached_0'}, {'key0': 'cached_1'}, None])
    >>> dataset = DatasetCachedUID(dataset_uids, ('uid',), '/path/to/cache', '1.0', 'demo')
    >>> dataset[0]

    `uid1` cannot be located, so `dataset_to_cache[0]` will computed and saved as .hdf5.
    the batch will have all the keys of `dataset_uids[0]` and `dataset_to_cache[0]`.`dataset_to_cache[0]`
    """

    def __init__(
        self,
        base_dataset: CoreDataset,
        base_dataset_uids: Sequence[str],
        dataset_to_cache: CoreDataset,
        path_to_cache_root: str,
        dataset_version: str,
        dataset_name: str,
        mode: Literal['r', 'w', 'a'] = 'r',
        read_data_fn: Callable[
            [
                str,
                NamedArg(Optional[Dict], 'context'),
            ],
            Batch,
        ] = read_data_h5,
        write_data_fn: Callable[[Dict, str], None] = write_data_h5,
        transform: Optional[Callable[[Batch], Batch]] = None,
        pre_transform: Optional[Callable[[Batch], Batch]] = None,
        nb_retry: int = 10,
        retry_sleep_time_sec: float = 1.0,
    ) -> None:
        """
        Args:
            base_dataset: the dataset holding UIDs
            dataset_to_cache: if the cache for the corresponding UID cannot be found
                the this dataset index will be calculated and cached using the UID
            nb_retry: indicate how many times to try processing an index. This is
                used to handle error in a multiprocessing context where processes
                may try to write to the same file
            retry_sleep_time_sec: sleep time between trials
            base_dataset_uids: use keys within a batch to be used as UID. This
                is useful to preprocess some datasets (e.g., segmentation) and
                keep the UID independent from the index. Beware, the UIDs needs
                to be unique!
        """

        super().__init__()
        self.base_dataset = base_dataset
        self.path_to_cache_root = path_to_cache_root
        self.mode = mode
        self.read_data_fn = read_data_fn
        self.write_data_fn = write_data_fn
        self.transform = transform
        self.pre_transform = pre_transform
        self.dataset_version = dataset_version
        self.dataset_name = dataset_name

        self.retry_sleep_time_sec = retry_sleep_time_sec
        self.nb_retry = nb_retry

        assert len(base_dataset_uids) > 0
        self.base_dataset_uids = base_dataset_uids
        self.dataset_to_cache = dataset_to_cache

        assert len(base_dataset_uids) >= 1
        assert len(base_dataset) == len(dataset_to_cache), (
            'base dataset (UIDs) and content (dataset_to_cache) must have the same size!'
            f' Got={len(base_dataset)} / {len(dataset_to_cache)}'
        )
        os.makedirs(self.path_to_cache_root, exist_ok=True)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _get_item_name(self, base_batch: Batch) -> str:
        uids = [safe_uid(base_batch[u]) for u in self.base_dataset_uids]
        uids = '_'.join(uids)
        local_file = os.path.join(self.path_to_cache_root, f'{self.dataset_name}-{uids}.h5')
        return local_file

    def _reprocess_caches_index(self, local_file: str, index: int, context: Optional[Dict] = None) -> None:
        # reprocess the index and cache the result
        logger.info(f'reprocessing index={index}')
        assert self.mode == 'a' or self.mode == 'w', f'invalid mode, cannot write! mode={self.mode}'
        batch = self.dataset_to_cache.__getitem__(index, context)
        assert batch is not None
        batch['dataset_cached_version'] = self.dataset_version

        if self.pre_transform is not None:
            batch = self.pre_transform(batch)

        self.write_data_fn(batch, local_file)

    def _get_item(self, base_batch: Batch, index: int, context: Optional[Dict] = None) -> Batch:
        local_file = self._get_item_name(base_batch)
        h5_valid = False
        if os.path.exists(local_file):
            try:
                with h5py.File(local_file) as f:
                    h5_version = f['dataset_cached_version'][()].decode()  # beware byte != str
                    if h5_version == self.dataset_version:
                        h5_valid = True
            except OSError:
                # H5 is invalid or corrupted, we need to
                # recalculate the cached data
                logger.exception(f'{local_file} is not valid!')
            except KeyError:
                logger.exception(f'{local_file} missing key `dataset_cached_version`!')

        if not h5_valid:
            self._reprocess_caches_index(local_file=local_file, index=index, context=context)

        # here the data MUST be valid!
        try:
            batch = self.read_data_fn(local_file, context=context)
        except OSError:
            # maybe there was a problem during the creation of the data. H5 is valid
            # but NOT one data field so reprocess it just in case.
            self._reprocess_caches_index(local_file=local_file, index=index, context=context)
            batch = self.read_data_fn(local_file, context=context)

        batch = {**base_batch, **batch}
        if self.transform is not None:
            batch = self.transform(batch)

        return batch

    def __getitem__(self, index: int, context: Optional[Dict] = None) -> Optional[Batch]:
        base_batch = self.base_dataset.__getitem__(index, context)
        if base_batch is None:
            return None

        for i in range(self.nb_retry):
            try:
                batch = self._get_item(base_batch, index, context)
                return batch
            except OSError as e:
                logger.info(
                    f'_get_item exception: {e}. Possibly reading or writing to the same '
                    f'file from different processes! Retry={i}, index={index}'
                )
                time.sleep(self.retry_sleep_time_sec)

        raise RuntimeError(f'Could not process index={index}')
