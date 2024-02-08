from flexdat import DatasetConcatenate, DatasetPath


def test_concatenate_3():
    d0 = DatasetPath(
        [
            'L0',
            'L1',
        ]
    )

    d1 = DatasetPath(
        [
            'L2',
            'L3',
            'L4',
        ]
    )

    d2 = DatasetPath(
        [
            'L5',
        ]
    )

    dataset = DatasetConcatenate({'d0': d0, 'd1': d1, 'd2': d2}, transform=lambda batch: batch['path'])
    assert len(dataset) == 6
    values = [dataset[i] for i in range(len(dataset))]
    assert values == ['L0', 'L1', 'L2', 'L3', 'L4', 'L5']

    dataset = DatasetConcatenate({'d0': d0, 'd1': d1, 'd2': d2})
    batch = dataset[5]
    assert batch['dataset_name'] == 'd2'
