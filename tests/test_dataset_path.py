from flexdat import DatasetPath


def test_dataset_path_with_metadata():
    paths = ['/path/1', '/path/2', '/path/3']
    doses = [1.0, 0.1, 0.5]
    dataset = DatasetPath(paths, dose=doses)
    assert len(dataset) == 3

    batch = dataset[2]
    assert len(batch) == 2
    assert batch['dose'] == 0.5
