from flexdat import DatasetDict, DatasetMerge


def test_dataset_merge():
    dataset_1 = DatasetDict([{'v0': 0, 'v1': 1}, {'v0': 10, 'v1': 11}])
    dataset_2 = DatasetDict([{'v2': 100, 'v3': 101}, {'v2': 110, 'v3': 111}])
    dataset = DatasetMerge([dataset_1, dataset_2])
    assert len(dataset) == 2
    assert dataset[1]['v2'] == 110


def test_dataset_merge_with_prefix():
    dataset_1 = DatasetDict([{'v0': 0, 'v1': 1}, {'v0': 10, 'v1': 11}])
    dataset_2 = DatasetDict([{'v2': 100, 'v3': 101}, {'v2': 110, 'v3': 111}])
    dataset = DatasetMerge([dataset_1, dataset_2], key_prefix=('dat0_', 'dat1_'))
    assert len(dataset) == 2
    assert dataset[1]['dat1_v2'] == 110
