from typing import Union

import numpy as np


def bytes2human(n: Union[int, float]) -> str:
    """
    Format large number of bytes into readable string for a human

    Examples:

        >>> bytes2human(10000)
        '9.8K'

        >>> bytes2human(100001221)
        '95.4M'

    """
    # http://code.activestate.com/recipes/578019
    # >>> bytes2human(10000)
    # '9.8K'
    # >>> bytes2human(100001221)
    # '95.4M'
    symbols = ('K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    prefix = {}
    for i, s in enumerate(symbols):
        prefix[s] = 1 << (i + 1) * 10
    for s in reversed(symbols):
        if n >= prefix[s]:
            value = float(n) / prefix[s]
            return '%.1f%s' % (value, s)
    return '%.2f' % n


def is_dtype_signed(a: np.dtype) -> bool:
    """
    Return True if an array type is signed
    """
    return a in (
        int,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        float,
        np.float128,
        np.float64,
        np.float32,
        np.float16,
    )


def is_dtype_integer(a: np.dtype) -> bool:
    """
    Return True if type should be considered as integers

    E.g., for resampling an image, what interpolator should we use?
    """
    return a in (int, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64, bool)


def is_dtype_number(a: np.dtype) -> bool:
    """
    Return True if type should be considered as numbers
    """
    return a in (
        int,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        float,
        np.float128,
        np.float256,
        np.float64,
        np.float32,
        np.float16,
        bool,
    )


def np_array_type_clip(v: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """
    Prevent wrap around if incorrectly casting int->uint

    The incorrect data range WILL be clipped

    >>> list(np_array_type_clip(np.asarray([1.0, 2.0, -1.0]), dtype=np.uint8))
    [1, 2, 0]
    """
    if not is_dtype_integer(dtype):
        # no clipping needed for floating types
        return v.astype(dtype)

    info = np.iinfo(dtype)
    return np.clip(v, info.min, info.max).astype(dtype)
