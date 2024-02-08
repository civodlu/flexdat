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


def is_numpy_integer_array(a: np.ndarray) -> bool:
    """
    Return True if an array type is should be considered as integers

    E.g., for resampling an image, what interpolator should we use?
    """
    assert isinstance(a, np.ndarray)
    return a.dtype in (int, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, bool)
