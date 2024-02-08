from flexdat import DatasetPath, DatasetResizeVirtual


def test_virtual_resize():
    paths = ['/path/1', '/path/2', '/path/3']
    dataset = DatasetPath(paths)
    dataset = DatasetResizeVirtual(dataset, size=10)
    assert dataset[3]['path'] == '/path/1'
    assert len(dataset) == 10
