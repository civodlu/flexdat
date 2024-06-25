from flexdat import DatasetPath, DatasetResizeVirtual


def test_virtual_resize():
    paths = ['/path/1', '/path/2', '/path/3']
    dataset = DatasetPath(paths)
    dataset = DatasetResizeVirtual(dataset, size=10)
    assert dataset[3]['path'] in paths
    assert len(dataset) == 10


def test_virtual_resize_multiple_datasets_sequential_sampling():
    paths_1 = ['/path/1', '/path/2', '/path/3']
    dataset_1 = DatasetPath(paths_1)

    paths_2 = ['/path/4', '/path/5']
    dataset_2 = DatasetPath(paths_2)

    dataset = DatasetResizeVirtual([dataset_1, dataset_2], size=50, sampling_mode='sequential')
    paths = [paths_1, paths_2]
    assert len(dataset) == 50

    paths_results = [set(), set()]
    for i in range(50):
        b = dataset[i]
        expected_dataset = i % 2
        assert b['path'] in paths[expected_dataset]
        paths_results[expected_dataset].add(b['path'])

    # make sure we sampled EVERYTHING
    assert paths_results[0] == set(paths[0])
    assert paths_results[1] == set(paths[1])
