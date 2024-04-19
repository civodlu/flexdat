import os
import pickle as pkl
import platform
import time

import pytest

from flexdat import DatasetCacheMemcached, DatasetImageFolder, DatasetPath
from flexdat.dataset_cached_memcached import encode_batch_pkl, encode_batch_pkl_lz4
from flexdat.utils import bytes2human


@pytest.mark.skipif(platform.system() != 'Linux', reason='Linux only!')
def test_dataset_memcached():
    encoders = [None, encode_batch_pkl, encode_batch_pkl_lz4]

    decoders = [None, encode_batch_pkl, encode_batch_pkl_lz4]

    for dat_n, (encoder, decoder) in enumerate(zip(encoders, decoders)):
        # requires memcached utility installed (sudo apt install memcached)
        # requires `memcached -I 500m -m 25000m -p 11212` to be started
        here = os.path.abspath(os.path.dirname(__file__))
        path = os.path.join(here, 'resource/nifti_01')
        dataset = DatasetPath([path])
        dataset = DatasetImageFolder(
            dataset
        )  # requires the correct memcached starting options, else no caching happens due to large size of the data
        dataset = DatasetCacheMemcached(
            dataset,
            # host=('localhost', 11212),
            database_name=f'test_dataset_cached_{dat_n}',
            key_expiry_time_sec=10,
            batch_encoder=encoder,
            batch_decoder=decoder,
        )
        assert len(dataset) == 1

        times = []
        for i in range(5):
            time_start = time.perf_counter()
            batch = dataset[0]
            time_end = time.perf_counter()

            times.append(time_end - time_start)
            batch_serialized = pkl.dumps(batch)
            batch_size = bytes2human(len(batch_serialized))
            print('batch_size=', batch_size)

        print('Times=', times, 'encoder=', encoder)
