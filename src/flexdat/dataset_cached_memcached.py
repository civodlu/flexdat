import pickle as pkl
from typing import Any, Callable, Dict, Optional

from .dataset import CoreDataset
from .types import Batch


def encode_batch_pkl_lz4(batch: Batch) -> bytes:
    from lz4.frame import compress

    batch_encoded = pkl.dumps(batch)
    batch_compressed: bytes = compress(batch_encoded)
    return batch_compressed


def encode_batch_pkl(batch: Batch) -> bytes:
    batch_encoded = pkl.dumps(batch)
    return batch_encoded


def decode_batch_pkl_lz4(compressed_bytes: bytes) -> Batch:
    from lz4.frame import decompress

    decompressed_bytes = decompress(compressed_bytes)
    batch: Batch = pkl.loads(decompressed_bytes)
    return batch


def decode_batch_pkl(compressed_bytes: bytes) -> Batch:
    batch: Batch = pkl.loads(compressed_bytes)
    return batch


class DatasetCacheMemcached(CoreDataset):
    """
    Cache the dataset in an in-memory key value store

    Sometimes, data preprocessing or network can be extremely costly. Instead,
    servers are often over-capacitated in terms of RAM. This dataset caches dataset in-memory.

    However, when using multiprocessing (or running multiple experiments using the same dataset),
    the dataset will be copied for each process. To avoid this, we need a remote
    process that will hold centrally all the datasets and share only the requested data to
    the worker process (i.e., hence memcached).

    Standard option for memcached:
        memcached -I 500m -m 25000m -p 11211

    With `-I` for the max size of an object in Mb and `-m` the maximum store size in Mb.

    Beware:
        * it is possible a memcached service is already running in the background!
          Meaning that options are not taken into account. The default maximum object size
          is 1 MB. In that case, the service should be stopped and restarted (`service memcached stop`)
        * unfortunately the buffer is not directly shared, it has to be serialized
          which can make large data share expensive (e.g., ~90MB -> 0.1s total access time)
    """

    def __init__(
        self,
        base_dataset: CoreDataset,
        database_name: str,
        host: Any = ('localhost', 11211),
        no_delay: bool = True,
        key_expiry_time_sec: int = 5 * 60 * 60,
        batch_encoder: Optional[Callable[[Optional[Batch]], bytes]] = None,
        batch_decoder: Optional[Callable[[bytes], Optional[Batch]]] = None,
        **db_connect_kwargs: Any,
    ) -> None:
        super().__init__()
        self.base_dataset = base_dataset
        self.database_name = database_name
        self.host = host
        self.key_expiry_time_sec = key_expiry_time_sec
        self.batch_encoder = batch_encoder
        self.batch_decoder = batch_decoder

        import pymemcache
        import pymemcache.client.base

        self.db_client = pymemcache.client.base.Client(
            host, no_delay=no_delay, serde=pymemcache.serde.PickleSerde(), **db_connect_kwargs
        )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int, context: Optional[Dict] = None) -> Optional[Batch]:
        key_name = f'{self.database_name}_{index}'
        batch_encoding = self.db_client.get(key_name, default=None)
        if batch_encoding is None:
            # the value is not present in the database, calculate it and store it
            batch = self.base_dataset.__getitem__(index, context)
            if self.batch_encoder is not None:
                batch_encoded = self.batch_encoder(batch)
                self.db_client.set(key_name, batch_encoded, expire=self.key_expiry_time_sec)
            else:
                self.db_client.set(key_name, batch, expire=self.key_expiry_time_sec)
            return batch

        if self.batch_decoder is not None:
            # time_start = time.perf_counter()
            batch = self.batch_decoder(batch_encoding)
            # time_end = time.perf_counter()
            # print('DecodingTime=', time_end - time_start)
        else:
            batch = batch_encoding
        return batch
