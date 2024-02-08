from flexdat import DatasetPath, DatasetTransform


def test_subset():
    paths = ['/path/1', '/path/2', '/path/3']
    dataset = DatasetPath(paths)

    def post_fix(batch):
        batch['path'] = batch['path'] + '/POSTFIX'
        return batch

    dataset = DatasetTransform(dataset, post_fix)
    assert len(dataset) == 3
    assert dataset[0]['path'] == '/path/1/POSTFIX'
