import time
from functools import partial

import numpy as np

from flexdat import DatasetDict
from flexdat.dataset_cached_h5 import SamplerH5
from flexdat.dataset_cached_pool import DatasetCachedPool
from flexdat.sampling import CoordinateSamplerBlock


def transform_add_value(batch, name, values, wait_time_sec=0.0):
    time.sleep(wait_time_sec)
    if name not in values:
        values[name] = 1
    else:
        values[name] += 1
    batch[name] = 'done'
    print(f'{name} called {values[name]} times')
    return batch


def make_array(shape, value):
    return np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + value


def test_cached_pool_dataset_serial():
    sample_1 = {
        'data_A': make_array([30, 31, 32], 10),
        'data_B': make_array([30, 31, 32], 20),
    }

    sample_2 = {
        'data_A': make_array([30, 31, 32], 30),
        'data_B': make_array([30, 31, 32], 40),
    }

    sampler = SamplerH5(coordinate_sampler=CoordinateSamplerBlock(block_shape=(4, 5, 6)))
    samples = [sample_1, sample_2]
    dataset = DatasetDict(samples)

    values = {}
    nb_number_of_reuse_per_entry = 3
    cached_dataset = DatasetCachedPool(
        dataset,
        cache_size=1,
        number_of_reuse_per_entry=nb_number_of_reuse_per_entry,
        pre_transform=partial(transform_add_value, name='pre_transform', values=values),
        transform=partial(transform_add_value, name='transform', values=values),
        sampler=sampler,
        num_background_workers=0,
        with_debug_info=True,
    )

    base_indices = set()
    for i in range(10):
        batch = cached_dataset[0]
        assert values['pre_transform'] == i // nb_number_of_reuse_per_entry + 1
        assert values['transform'] == i + 1

        base_index = batch['dataset_cached_pool_index']
        base_indices.add(base_index)
        assert batch['data_A'].shape == (1, 1, 4, 5, 6)
        assert batch['data_B'].shape == (1, 1, 4, 5, 6)

        sample = dataset[base_index]
        min_index = np.asarray(batch['sampling_indices_min_zyx'][0])
        max_index = np.asarray(batch['sampling_indices_max_zyx'][0])

        # the sampling MUST be synchronized between the 2 features
        sl = tuple(slice(min_index[i], max_index[i] + 1) for i in range(3))
        assert (batch['data_A'].squeeze() == sample['data_A'][sl]).all()
        assert (batch['data_B'].squeeze() == sample['data_B'][sl]).all()

    assert len(base_indices) == 2


def test_cached_pool_dataset_async():
    sample_1 = {
        'data_A': make_array([30, 31, 32], 10),
    }

    dataset = DatasetDict([sample_1])

    values = {}
    nb_number_of_reuse_per_entry = 3
    cached_dataset = DatasetCachedPool(
        dataset,
        cache_size=1,
        number_of_reuse_per_entry=nb_number_of_reuse_per_entry,
        pre_transform=partial(transform_add_value, name='pre_transform', values=values, wait_time_sec=2.0),
        num_background_workers=1,
        with_debug_info=True,
    )

    # first call will trigger the pre_transform in background
    # we expect None as the result is not ready yet
    batch = cached_dataset[0]
    assert batch is None

    time.sleep(5.0)  # wait for the background thread to finish
    batch = cached_dataset[0]
    assert batch is not None
    assert batch['dataset_cached_pool_debug_reuse_count'] == 1

    batch = cached_dataset[0]
    assert batch is not None
    assert batch['dataset_cached_pool_debug_reuse_count'] == 2

    batch = cached_dataset[0]
    assert batch is not None
    assert batch['dataset_cached_pool_debug_reuse_count'] == 3

    # refresh sample in background but return stale sample for nor
    batch = cached_dataset[0]
    assert batch is not None
    assert batch['dataset_cached_pool_debug_reuse_count'] == 4

    # wait for the background thread to finish
    time.sleep(5.0)

    # sample is fresh
    batch = cached_dataset[0]
    assert batch is not None
    assert batch['dataset_cached_pool_debug_reuse_count'] == 1
