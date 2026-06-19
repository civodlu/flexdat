from concurrent.futures import Future, ThreadPoolExecutor
from copy import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from flexdat.dataset import Batch, CoreDataset, NonDeterministicDataset
from flexdat.dataset_cached_h5 import Sampler


class SerialFuture:
    """A Future that completed synchronously in the calling thread."""

    def __init__(self, result: Any):
        self._result = result

    def done(self) -> bool:
        return True

    def result(self) -> Any:
        return self._result


class SerialExecutor:
    """Drop-in for ThreadPoolExecutor that runs tasks immediately in the same thread."""

    def submit(self, fn: Callable, *args, **kwargs) -> SerialFuture:
        return SerialFuture(fn(*args, **kwargs))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


@dataclass
class CacheSlot:
    reuse_count: int
    base_index: Optional[int] = None
    batch: Optional[Batch] = None
    future: Optional[Any] = None  # Future | SerialFuture | None


class DatasetCachedPool(NonDeterministicDataset):
    """
    This dataset caches entries from another dataset to perform costly operations (e.g., augmentation, registration)

    Each entries may be reused a fixed number of times before being refreshed from the base dataset.

    This is designed to be used with `DatasetCachedH5` or similar.

    When `num_background_workers > 0`, expired cache slots are reloaded in the background while the
    current (stale) entry continues to be served, so the pipeline is never blocked.
    Set `num_background_workers=0` for serial/synchronous mode (easier debugging).
    """

    def __init__(
        self,
        dataset: CoreDataset,
        cache_size: int = 30,
        number_of_reuse_per_entry: int = 10,
        pre_transform: Optional[Callable[[Batch], Batch]] = None,
        transform: Optional[Callable[[Batch], Batch]] = None,
        sampler: Optional[Sampler] = None,
        default_context: Dict = {
            'dataset_h5_disable_sampler': True,
            'dataset_h5_disable_transform': True,
        },
        num_background_workers: int = 2,
        with_debug_info: bool = False,
    ):
        """
        Args:
            dataset: the base dataset to cache
            cache_size: the number of entries to cache
            number_of_reuse_per_entry: the number of times to reuse an entry before reloading
            pre_transform: a function to apply to the batch before caching
            transform: a function to apply to the batch after caching
            sampler: a function to apply to the batch after caching and before transform
            default_context: the context to use when loading from the base dataset
            num_background_workers: number of threads for background reloading.
                0 = serial/synchronous mode (runs in same thread, easier to debug).
                >0 = async mode (serves stale entry while reloading in background).
        """
        super().__init__()
        self.dataset = dataset
        self.cache_size = cache_size
        self.number_of_reuse_per_entry = number_of_reuse_per_entry
        self.pre_transform = pre_transform
        self.transform = transform
        self.cache: List[CacheSlot] = [CacheSlot(reuse_count=number_of_reuse_per_entry) for _ in range(cache_size)]
        self.sampler = sampler
        self.default_context = default_context
        self.executor = (
            ThreadPoolExecutor(max_workers=num_background_workers) if num_background_workers > 0 else SerialExecutor()
        )
        self.with_debug_info = with_debug_info

        if len(dataset) == 0:
            raise ValueError('Dataset is empty!')

    def _load_entry(self) -> Tuple[int, Batch]:
        base_index = np.random.randint(0, len(self.dataset))
        # make sure we copy the default_context, otherwise the mod
        batch = self.dataset.__getitem__(base_index, copy(self.default_context))
        if batch is not None and self.pre_transform is not None:
            batch = self.pre_transform(batch)
        return base_index, batch

    def __getitem__(self, index: int, context: Optional[Dict] = None) -> Optional[Batch]:
        assert index < self.cache_size
        slot = self.cache[index]

        # Collect completed reload (immediate in serial mode, may be pending in async)
        if slot.future is not None and slot.future.done():
            slot.base_index, slot.batch = slot.future.result()
            slot.reuse_count = 0
            slot.future = None

        if slot.reuse_count >= self.number_of_reuse_per_entry and slot.future is None:
            # Fire reload; serial mode completes it now, async mode runs it in background
            slot.future = self.executor.submit(self._load_entry)
            # In serial mode, collect immediately; in async mode, this won't be done yet
            if slot.future.done():
                slot.base_index, slot.batch = slot.future.result()
                slot.reuse_count = 0
                slot.future = None
            elif not slot.batch:
                # Cold start in async mode: no data yet, return None so caller can discard and retry
                return None

        slot.reuse_count += 1
        if slot.batch is None:
            return None
        batch = copy(slot.batch)

        if self.sampler is not None:
            batch = self.sampler(batch, context)

            # copy keys not present
            for key, value in slot.batch.items():
                if key not in batch:
                    if isinstance(value, bytes):
                        value = value.decode()
                    batch[key] = value
        if self.transform is not None:
            batch = self.transform(batch)

        if self.with_debug_info:
            batch['dataset_cached_pool_debug_reuse_count'] = slot.reuse_count
            batch['dataset_cached_pool_index'] = slot.base_index

        return batch

    def __len__(self) -> int:
        return self.cache_size
