from typing import Sequence

import numpy as np
from torch.utils.data.dataloader import default_collate

from flexdat import DatasetPath
from flexdat.dataset_multi_samples import DatasetMultiSample


def assert_result(paths, nb_samples):
    dataset = DatasetPath(paths)
    dataset = DatasetMultiSample(dataset, nb_samples=nb_samples, collate_fn=default_collate)

    assert len(dataset) == len(paths) ** nb_samples
    path_samples = []
    for i in range(len(dataset)):
        p = dataset[i]['path']
        path_samples.append(p)

    from collections import Counter

    counts = Counter(np.concatenate(path_samples))
    assert len(counts) == len(paths)
    expected_v = next(iter(counts.values()))
    for k, v in counts.items():
        assert k in paths
        assert v == expected_v


def test_dataset_multi_samples():
    assert_result(['/path/1', '/path/2'], nb_samples=3)
    assert_result(['/path/1', '/path/2', '/path/3'], nb_samples=2)
    assert_result(['/path/1', '/path/2', '/path/3'], nb_samples=3)
    assert_result(['/path/1', '/path/2', '/path/3', '/path/4'], nb_samples=10)
