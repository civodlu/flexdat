from flexdat import CoreDataset, DatasetSafe
from flexdat.dataset_safe import DatasetExceptionDiscard


class DatasetRaiseException(CoreDataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return 3

    def __getitem__(self, index, context=None):
        if index == 0:
            return {'key0': 'value0'}
        elif index == 1:
            raise DatasetExceptionDiscard('Should be intercepted and discarded!')
        else:
            raise RuntimeError('Should be re-raised!')


def test_exception_caught():
    dataset = DatasetRaiseException()
    dataset = DatasetSafe(dataset)

    batch = dataset[0]
    assert len(batch) == 1
    assert batch['key0'] == 'value0'

    batch = dataset[1]
    assert batch is None

    exception_raise = False
    try:
        batch = dataset[2]
    except RuntimeError:
        exception_raise = True

    assert exception_raise
