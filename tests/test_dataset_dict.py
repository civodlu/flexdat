from flexdat import DatasetDict


def test_dataset_dict():
    dataset = DatasetDict([{'v0': 0, 'v1': 1}, {'v0': 10, 'v1': 11}])
    assert len(dataset) == 2
    assert dataset[0]['v1'] == 1
    assert dataset[0]['v0'] == 0
    assert dataset[1]['v1'] == 11
    assert dataset[1]['v0'] == 10


def test_dataset_dict_append():
    dataset = DatasetDict()
    assert len(dataset) == 0
    dataset.append({'v0': 0, 'v1': 1})
    dataset.append({'v0': 10, 'v1': 11})

    assert len(dataset) == 2
    assert dataset[0]['v1'] == 1
    assert dataset[0]['v0'] == 0
    assert dataset[1]['v1'] == 11
    assert dataset[1]['v0'] == 10
