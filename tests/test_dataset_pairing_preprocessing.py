from flexdat import DatasetPaired, DatasetPairingPreprocessor, DatasetPath
from flexdat.sampler_pairing import PairingSamplerRandom


def test_metadata_injector():
    dataset = DatasetPath(['location_1_full', 'location_2_full', 'location_1_low', 'location_2_low'])

    def metadata_fn(batches):
        return {'data': [b['path'] for b in batches]}

    associations = [(1, 3), (0, 2)]
    dataset = DatasetPairingPreprocessor(dataset, associations, metadata_fn)
    assert len(dataset) == 4

    dataset = DatasetPaired(
        dataset, associations, pairing_sampler=PairingSamplerRandom(pairing_key_prefix=('full_', 'low_'))
    )
    assert len(dataset) == 2

    batch = dataset[1]
    assert len(batch) == 4
    assert batch['full_path'] == 'location_1_full'
    assert batch['low_path'] == 'location_1_low'
    assert batch['full_data'] == ['location_1_full', 'location_1_low']
    assert batch['low_data'] == ['location_1_full', 'location_1_low']

    batch = dataset[0]
    assert len(batch) == 4
    assert batch['full_path'] == 'location_2_full'
    assert batch['low_path'] == 'location_2_low'
    assert batch['full_data'] == ['location_2_full', 'location_2_low']
    assert batch['low_data'] == ['location_2_full', 'location_2_low']


test_metadata_injector()
