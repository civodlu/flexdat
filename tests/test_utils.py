import numpy as np

from flexdat.utils import bytes2human, is_dtype_integer, is_dtype_signed


def test_bytes2human():
    assert bytes2human(10000) == '9.8K'
    assert bytes2human(100001221) == '95.4M'


def test_is_numpy_integer_array():
    assert is_dtype_integer(np.zeros([4], dtype=np.uint8).dtype)
    assert is_dtype_integer(np.zeros([4], dtype=np.int8).dtype)
    assert not is_dtype_integer(np.zeros([4], dtype=np.float16).dtype)


def test_is_numpy_signed():
    assert not is_dtype_signed(bool)
    assert not is_dtype_signed(np.uint16)
    assert is_dtype_signed(np.int16)
    assert is_dtype_signed(int)
    assert not is_dtype_signed(str)
