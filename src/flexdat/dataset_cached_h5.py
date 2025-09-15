import logging
import os
import time
from typing import Callable, Dict, Literal, Optional

import h5py
from mypy_extensions import DefaultNamedArg, NamedArg

from .dataset import CoreDataset
from .sampling import SamplerH5
from .types import Batch
from .utils_h5 import (
    COMPLETION_FLAG_NAME,
    COMPLETION_FLAG_VALUE,
    THIS_BATCH_IS_NONE_FLAG_NAME,
    read_data_h5,
    write_data_h5,
)

logger = logging.getLogger(__name__)

Sampler = Callable[[h5py.File], Batch]


def is_hdf5_valid(local_file: str, dataset_version: str) -> bool:
    """
    Return True if the file is valid
    """
    if os.path.exists(local_file):
        # file exist, can be opened and a version was recorded
        try:
            with h5py.File(local_file) as f:
                if THIS_BATCH_IS_NONE_FLAG_NAME in f:
                    # it is a valid file but considered "None"
                    return True

                h5_version = f['dataset_version'][()].decode()  # beware byte != str
                if h5_version != dataset_version:
                    return False

                h5_name = f['dataset_name'][()].decode()  # beware byte != str
                if len(h5_name) == 0:
                    return False

                if COMPLETION_FLAG_NAME not in f:
                    # we are expecting a completion flag to signal the file
                    # was fully processed and written to disk even in the case
                    # where a process was terminated
                    return False

                completion_flag_found = f[COMPLETION_FLAG_NAME][()].decode()
                if completion_flag_found != COMPLETION_FLAG_VALUE:
                    return False

        except OSError:
            # H5 is invalid or corrupted, we need to
            # recalculate the cached data
            logger.exception(f'{local_file} is not valid!')
            return False

        except KeyError:
            logger.exception(f'{local_file} missing key `dataset_version`!')
            return False

        return True
    else:
        return False


class DatasetCachedH5(CoreDataset):
    """
    Cache a whole dataset using H5.

    The purpose is to cache dataset that have very long computations (e.g., whole body
    segmentation denoising or reconstruction). Each `base_dataset` index will be serialized
    as a single H5 dataset. This is designed for heavy datasets (e.g., full 3D volumes).

    The default `write_data_h5` supports data compression and chunking so that data can
    be partially loaded (e.g., if we only want a 64x64x64 sub-block of data)

    Sub-volumes can be loaded using the `sampler` parameter.

    For example:

    >>> paths = ['/path/1', '/path/2']
    >>> dataset = DatasetPath(paths)
    >>> dataset = DatasetImageReader(dataset, path_reader=partial(path_reader_dicom, image_namer=lambda h: h['Modality']))
    >>> dataset = DatasetCachedH5(dataset, 'path/to/cache', 'dataset/name', 'dataset/version', 'a')
    >>> batch = dataset[0]

    Notes:
        * with `context['dataset_h5_disable_sampler'] = False`, the sampling will not be applied
          and the dataset will be loaded whole
    """

    def __init__(
        self,
        base_dataset: CoreDataset,
        path_to_cache_root: str,
        dataset_version: str,
        dataset_name: str,
        mode: Literal['r', 'w', 'a'] = 'r',
        new_dataset_size: Optional[None] = None,
        read_data_fn: Callable[
            [
                str,
                NamedArg(Optional[Sampler], 'sampler'),
                NamedArg(Optional[Dict], 'context'),
            ],
            Batch,
        ] = read_data_h5,  # type: ignore
        write_data_fn: Callable[[Dict, str], None] = write_data_h5,
        transform: Optional[Callable[[Batch], Batch]] = None,
        pre_transform: Optional[Callable[[Batch], Batch]] = None,
        sampler: Optional[Sampler] = SamplerH5(),
        nb_retry: int = 10,
        retry_sleep_time_sec: float = 1.0,
    ) -> None:
        """
        Args:
            nb_retry: indicate how many times to try processing an index. This is
                used to handle error in a multiprocessing context where processes
                may try to write to the same file
            retry_sleep_time_sec: sleep time between trials
        """

        super().__init__()
        self.base_dataset = base_dataset
        self.path_to_cache_root = path_to_cache_root
        self.mode = mode
        self.new_dataset_size = new_dataset_size
        self.read_data_fn = read_data_fn
        self.write_data_fn = write_data_fn
        self.transform = transform
        self.pre_transform = pre_transform
        self.dataset_version = dataset_version
        self.sampler = sampler
        self.dataset_name = dataset_name
        assert len(dataset_name) > 1, 'invalid name!'

        self.retry_sleep_time_sec = retry_sleep_time_sec
        self.nb_retry = nb_retry

        os.makedirs(self.path_to_cache_root, exist_ok=True)

    def __len__(self) -> int:
        if self.new_dataset_size is None:
            return len(self.base_dataset)
        else:
            return self.new_dataset_size

    def _get_item_name(self, index: int) -> str:
        local_file = os.path.join(self.path_to_cache_root, f'{self.dataset_name}-{index}.h5')
        return local_file

    @staticmethod
    def is_batch_none(batch: Batch) -> bool:
        return THIS_BATCH_IS_NONE_FLAG_NAME in batch

    def _reprocess_caches_index(self, local_file: str, index: int, context: Optional[Dict] = None) -> None:
        # reprocess the index and cache the result
        logger.info(f'reprocessing index={index}')
        assert self.mode == 'a' or self.mode == 'w', f'invalid mode, cannot write! mode={self.mode}'
        batch = self.base_dataset.__getitem__(index, context)
        batch_is_none = False
        if batch is None:
            batch_is_none = True
            batch = {THIS_BATCH_IS_NONE_FLAG_NAME: True}

        batch['dataset_version'] = self.dataset_version

        if self.pre_transform is not None and not batch_is_none:
            batch = self.pre_transform(batch)

        self.write_data_fn(batch, local_file)

    def _get_item(self, index: int, context: Optional[Dict] = None) -> Optional[Batch]:
        local_file = self._get_item_name(index)
        h5_valid = is_hdf5_valid(local_file, dataset_version=self.dataset_version)
        if not h5_valid:
            self._reprocess_caches_index(local_file=local_file, index=index, context=context)

        # for evaluation purposes, we may want to process the whole data
        # so discard the sampler by using the context
        dataset_h5_disable_sampler = None
        if context is not None:
            dataset_h5_disable_sampler = context.get('dataset_h5_disable_sampler')
        sampler = self.sampler if not dataset_h5_disable_sampler else None

        # here the data MUST be valid!
        try:
            batch = self.read_data_fn(local_file, sampler=sampler, context=context)
            if DatasetCachedH5.is_batch_none(batch):
                return None
        except OSError:
            # maybe there was a problem during the creation of the data. H5 is valid
            # but NOT one data field so reprocess it just in case.
            self._reprocess_caches_index(local_file=local_file, index=index, context=context)
            batch = self.read_data_fn(local_file, sampler=sampler, context=context)

        if self.transform is not None:
            batch = self.transform(batch)

        return batch

    def __getitem__(self, index: int, context: Optional[Dict] = None) -> Optional[Batch]:
        for i in range(self.nb_retry):
            try:
                batch = self._get_item(index, context)
                return batch
            except OSError as e:
                logger.info(
                    f'_get_item exception: {e}. Possibly reading or writing to the same file from'
                    f' different processes! Retry={i}, index={index}'
                )
                time.sleep(self.retry_sleep_time_sec)

        raise RuntimeError(f'Could not process index={index}')
