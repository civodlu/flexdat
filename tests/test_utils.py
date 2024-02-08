import numpy as np

from flexdat.utils import bytes2human, is_numpy_integer_array


def test_bytes2human():
    assert bytes2human(10000) == '9.8K'
    assert bytes2human(100001221) == '95.4M'


def test_is_numpy_integer_array():
    assert is_numpy_integer_array(np.zeros([4], dtype=np.uint8))
    assert is_numpy_integer_array(np.zeros([4], dtype=np.int8))
    assert not is_numpy_integer_array(np.zeros([4], dtype=np.float16))
