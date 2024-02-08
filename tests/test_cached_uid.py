import tempfile

from utils import DatasetAccessCounter

from flexdat import DatasetCachedUID, DatasetDict


def test_cached_uid():
    dataset_uids = DatasetDict([{'uid': 'uid1'}, {'uid': 'uid2'}, {'uid': 'uid1'}])
    dataset_to_cache = DatasetDict([{'key0': 'cached_0'}, {'key0': 'cached_1'}, None])
    dataset_to_cache = DatasetAccessCounter(dataset_to_cache)

    with tempfile.TemporaryDirectory() as path_to_cache_root:
        # path_to_cache_root = '/tmp/experiments/'
        dataset = DatasetCachedUID(
            dataset_uids,
            ('uid',),
            dataset_to_cache,
            path_to_cache_root=path_to_cache_root,
            dataset_version='1.0',
            dataset_name='test_cached',
            mode='w',
        )

        assert len(dataset) == 3

        b = dataset[0]
        assert b['uid'] == 'uid1'
        assert b['dataset_cached_version'] == '1.0'
        assert b['key0'] == 'cached_0'

        # re-use the cache
        b = dataset[0]
        b = dataset[0]
        assert dataset_to_cache.counter[0] == 1

        b = dataset[1]
        assert b['uid'] == 'uid2'
        assert b['dataset_cached_version'] == '1.0'
        assert b['key0'] == 'cached_1'
        b = dataset[1]
        assert dataset_to_cache.counter[1] == 1

        # reused the cache of index `0`
        b = dataset[2]
        assert dataset_to_cache.counter[2] == 0
        assert b['uid'] == 'uid1'
        assert b['dataset_cached_version'] == '1.0'
        assert b['key0'] == 'cached_0'
        assert dataset_to_cache.counter[0] == 1  # it was cached so no recalculation occurred

        # new version
        dataset = DatasetCachedUID(
            dataset_uids,
            ('uid',),
            dataset_to_cache,
            path_to_cache_root=path_to_cache_root,
            dataset_version='1.1',
            dataset_name='test_cached',
            mode='w',
        )

        # version update, cache needs to be recalculated!
        b = dataset[0]
        assert dataset_to_cache.counter[0] == 2
