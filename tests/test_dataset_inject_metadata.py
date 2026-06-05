from flexdat import DatasetDict, DatasetInjectMetadata


def test_dataset_inject_metadata():
    metadata = {'id_0': {'value': 'v0'}, 'id_1': {'value': 'v1'}}
    dataset = DatasetDict([{'uid': 'id_1', 'value_2': 'v2_1'}, {'uid': 'id_0', 'value_2': 'v2_0'}])
    dataset = DatasetInjectMetadata(dataset, metadata, key='uid')
    assert len(dataset) == 2

    v0 = dataset[0]
    assert v0['value'] == 'v1'
    assert v0['value_2'] == 'v2_1'
    assert v0['uid'] == 'id_1'

    v0 = dataset[1]
    assert v0['value'] == 'v0'
    assert v0['value_2'] == 'v2_0'
    assert v0['uid'] == 'id_0'


def test_dataset_inject_metadata_missing_ok():
    metadata = {}
    dataset = DatasetDict([{'uid': 'id_1', 'value_2': 'v2_1'}, {'uid': 'id_0', 'value_2': 'v2_0'}])
    dataset = DatasetInjectMetadata(dataset, metadata, key='uid', raise_on_missing=False)
    assert len(dataset) == 2

    v0 = dataset[0]
    assert v0['value_2'] == 'v2_1'
    assert v0['uid'] == 'id_1'


def test_dataset_inject_metadata_missing_ko():
    metadata = {}
    dataset = DatasetDict([{'uid': 'id_1', 'value_2': 'v2_1'}, {'uid': 'id_0', 'value_2': 'v2_0'}])
    dataset = DatasetInjectMetadata(dataset, metadata, key='uid', raise_on_missing=True)
    assert len(dataset) == 2

    try:
        dataset[0]
        assert False, "Expected KeyError"
    except KeyError:
        pass


def test_dataset_inject_metadata_missing_ok_default_value():
    metadata = {}
    dataset = DatasetDict([{'uid': 'id_1', 'value_2': 'v2_1'}, {'uid': 'id_0', 'value_2': 'v2_0'}])
    dataset = DatasetInjectMetadata(
        dataset, metadata, key='uid', raise_on_missing=False, missing_metadata_value={'missing': 'YES'}
    )
    assert len(dataset) == 2

    v0 = dataset[0]
    assert v0['value_2'] == 'v2_1'
    assert v0['uid'] == 'id_1'
    assert v0['missing'] == 'YES'
