from typing import Any, Callable, Dict, Optional, Tuple

import h5py
import numpy as np
import torch

from .sampling import SamplerH5
from .types import Batch


def chunking_slice_fn(v: Any, max_dim: int = 16, max_slice: int = 8, max_time: int = 1) -> Optional[Tuple[int, ...]]:
    """
    Calculate the chunk shape from data
    """
    if isinstance(v, (np.ndarray, torch.Tensor)):
        if len(v.shape) == 1:
            return None
        elif len(v.shape) == 2:
            chunks = tuple(list(np.minimum(np.asarray(v.shape), max_dim)))
            return chunks
        elif len(v.shape) == 3:
            chunks = tuple([max_slice] + list(np.minimum(np.asarray(v.shape[1:]), max_dim)))
            return chunks
        elif len(v.shape) == 4:
            # if dim == 4, most likely dynamic series, usage will most likely be
            # time index independent (so should be == 1)
            chunks = tuple([max_time, max_slice] + list(np.minimum(np.asarray(v.shape[2:]), max_dim)))
            return chunks
        else:
            raise ValueError(f'unsupported chunking shape! Got={v.shape}')

    return None


def write_data_h5(
    data: Batch,
    path: str,
    compression: str = 'gzip',
    chunking_fn: Callable[[Any], Optional[Tuple[int, ...]]] = chunking_slice_fn,
) -> None:
    with h5py.File(path, 'w') as f:
        for name, value in data.items():
            if isinstance(value, torch.Tensor):
                value = value.numpy()
            chunks = chunking_fn(value) if chunking_fn is not None else None

            if isinstance(value, np.ndarray) and len(value.shape) >= 3:
                # 3D dataset: zip & chunks
                f.create_dataset(name, data=value, compression=compression, chunks=chunks)
            else:
                # any other data
                f.create_dataset(name, data=value)


def read_data_h5(
    path: str,
    sampler: Optional[SamplerH5] = None,
    context: Optional[Dict] = None,
) -> Batch:
    """
    Read data stored as H5. The may could be loaded partially if chunked.
    """
    with h5py.File(path, 'r') as f:
        if sampler is not None:
            batch = sampler(f, context=context)
        else:
            batch = {}

        for key in f.keys():
            if key not in batch:
                value = f[key][()]
                if isinstance(value, bytes):
                    value = value.decode()
                batch[key] = value

    return batch
