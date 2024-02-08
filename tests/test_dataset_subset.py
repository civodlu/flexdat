from flexdat import DatasetPath, DatasetSubset


def test_subset():
    paths = ['/path/1', '/path/2', '/path/3']
    dataset = DatasetPath(paths)
    dataset = DatasetSubset(dataset, indices=[1, 2])
    assert len(dataset) == 2
    assert dataset[0]['path'] == '/path/2'
