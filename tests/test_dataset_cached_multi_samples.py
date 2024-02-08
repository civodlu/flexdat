from utils import DatasetAccessCounter

from flexdat import DatasetCachedMultiSamples, DatasetPath


def test_cached_single_sample():
    dataset = DatasetPath(['path_0', 'path_1', 'path_2'])
    dataset = DatasetAccessCounter(dataset)
    counter = dataset.counter

    dataset = DatasetCachedMultiSamples(dataset, cache_size=1)

    # calculate new index
    assert len(counter) == 0
    assert dataset[0]['path'] == 'path_0'
    assert len(counter) == 1
    assert counter[0] == 1

    # re-use index
    assert dataset[0]['path'] == 'path_0'
    assert len(counter) == 1
    assert counter[0] == 1

    # calculate new index
    assert dataset[2]['path'] == 'path_2'
    assert len(counter) == 2
    assert counter[0] == 1
    assert counter[2] == 1

    # calculate new index
    assert dataset[0]['path'] == 'path_0'
    assert len(counter) == 2
    assert counter[0] == 2
    assert counter[2] == 1


def test_cached_2_samples():
    dataset = DatasetPath(['path_0', 'path_1', 'path_2'])
    dataset = DatasetAccessCounter(dataset)
    counter = dataset.counter

    dataset = DatasetCachedMultiSamples(dataset, cache_size=2)

    # calculate new index
    assert len(counter) == 0
    assert dataset[0]['path'] == 'path_0'
    assert len(counter) == 1
    assert counter[0] == 1

    # re-use index
    assert dataset[0]['path'] == 'path_0'
    assert len(counter) == 1
    assert counter[0] == 1

    # calculate new index
    assert dataset[2]['path'] == 'path_2'
    assert len(counter) == 2
    assert counter[0] == 1
    assert counter[2] == 1

    # re-use index
    assert dataset[0]['path'] == 'path_0'
    assert len(counter) == 2
    assert counter[0] == 1
    assert counter[2] == 1

    # calculate new index
    assert dataset[1]['path'] == 'path_1'
    assert len(counter) == 3
    assert counter[0] == 1
    assert counter[2] == 1
    assert counter[1] == 1

    # re-calculate out of cache index
    assert dataset[0]['path'] == 'path_0'
    assert len(counter) == 3
    assert counter[0] == 2
    assert counter[2] == 1
    assert counter[1] == 1
