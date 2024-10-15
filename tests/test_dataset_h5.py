import os
import tempfile
from functools import partial

import numpy as np

from flexdat import DatasetCachedH5, DatasetPaired, DatasetPairedList, DatasetPath
from flexdat.dataset_dicom import path_reader_dicom
from flexdat.dataset_image_reader import DatasetImageReader
from flexdat.sampler_pairing import (
    PairingSamplerEnumerate,
    PairingSamplerEnumerateNamed,
    PairingSamplerRandom,
)
from flexdat.sampling import CoordinateSamplerBlock, SamplerH5

here = os.path.abspath(os.path.dirname(__file__))


def test_basic_dataset_caching():
    with tempfile.TemporaryDirectory() as path:
        dataset = DatasetPath([os.path.join(here, 'resource/dicom_01')])
        dataset = DatasetImageReader(dataset, path_reader=partial(path_reader_dicom, image_namer=lambda h: h['Modality']))
        dataset = DatasetCachedH5(
            base_dataset=dataset, path_to_cache_root=path, dataset_name='dataset-test', dataset_version='1.0', mode='a'
        )

        b = dataset[0]
        assert os.path.exists(os.path.join(path, 'dataset-test-0.h5'))
        assert b['MR_voxels'].shape == (1, 3, 256, 192)
        indices = b['sampling_indices_min_zyx']
        assert len(indices) == 1
        index = indices[0]
        assert index == (index[0], None, None)

        indices = b['sampling_indices_max_zyx']
        assert len(indices) == 1
        index = indices[0]
        assert index == (index[0], None, None)


def test_paired_h5_sampler():
    with tempfile.TemporaryDirectory() as path:
        dataset = DatasetPath([os.path.join(here, 'resource/dicom_01')] * 2)
        dataset = DatasetImageReader(dataset, path_reader=partial(path_reader_dicom, image_namer=lambda h: h['Modality']))
        dataset = DatasetCachedH5(
            base_dataset=dataset, path_to_cache_root=path, dataset_name='dataset-test', dataset_version='1.0', mode='a'
        )

        dataset = DatasetPaired(dataset, [[0, 0]], pairing_sampler=PairingSamplerEnumerate())
        batch = dataset[0]

        # the sampler MUST be linked in paired dataset!
        assert batch['sample_0_sampling_indices_min_zyx'] == batch['sample_1_sampling_indices_min_zyx']
        assert batch['sample_0_sampling_indices_max_zyx'] == batch['sample_1_sampling_indices_max_zyx']

        # if this flag is present in the context, sampler should not be used!
        context = {'dataset_h5_disable_sampler': True}
        batch = dataset.__getitem__(0, context=context)
        assert batch['sample_1_MR_voxels'].shape == (60, 256, 192)
        assert batch['sample_0_MR_voxels'].shape == (60, 256, 192)


def test_paired_h5_sampler_named():
    with tempfile.TemporaryDirectory() as path:
        dataset = DatasetPath([os.path.join(here, 'resource/dicom_01')] * 2)
        dataset = DatasetImageReader(dataset, path_reader=partial(path_reader_dicom, image_namer=lambda h: h['Modality']))
        dataset = DatasetCachedH5(
            base_dataset=dataset, path_to_cache_root=path, dataset_name='dataset-test', dataset_version='1.0', mode='a'
        )

        dataset = DatasetPaired(dataset, [[0, 0]], pairing_sampler=PairingSamplerEnumerateNamed(('first_', 'second_')))
        batch = dataset[0]

        # the sampler MUST be linked in paired dataset!
        assert batch['first_sampling_indices_min_zyx'] == batch['second_sampling_indices_min_zyx']
        assert batch['first_sampling_indices_max_zyx'] == batch['second_sampling_indices_max_zyx']

        # if this flag is present in the context, sampler should not be used!
        context = {'dataset_h5_disable_sampler': True}
        batch = dataset.__getitem__(0, context=context)
        assert batch['second_MR_voxels'].shape == (60, 256, 192)
        assert batch['first_MR_voxels'].shape == (60, 256, 192)


def test_paired_h5_sampler_random():
    with tempfile.TemporaryDirectory() as path:
        dataset = DatasetPath([os.path.join(here, 'resource/dicom_01')] * 2)
        dataset = DatasetImageReader(dataset, path_reader=partial(path_reader_dicom, image_namer=lambda h: h['Modality']))
        dataset = DatasetCachedH5(
            base_dataset=dataset, path_to_cache_root=path, dataset_name='dataset-test', dataset_version='1.0', mode='a'
        )

        dataset = DatasetPaired(dataset, [[0, 0]], pairing_sampler=PairingSamplerRandom())
        batch = dataset[0]

        # the sampler MUST be linked in paired dataset!
        assert batch['p0_sampling_indices_min_zyx'] == batch['p1_sampling_indices_min_zyx']
        assert batch['p0_sampling_indices_max_zyx'] == batch['p1_sampling_indices_max_zyx']

        # if this flag is present in the context, sampler should not be used!
        context = {'dataset_h5_disable_sampler': True}
        batch = dataset.__getitem__(0, context=context)
        assert batch['p0_MR_voxels'].shape == (60, 256, 192)
        assert batch['p1_MR_voxels'].shape == (60, 256, 192)


def test_paired_list_h5_sampler():
    with tempfile.TemporaryDirectory() as path:
        dataset = DatasetPath([os.path.join(here, 'resource/dicom_01')] * 2)
        dataset = DatasetImageReader(dataset, path_reader=partial(path_reader_dicom, image_namer=lambda h: h['Modality']))
        dataset = DatasetCachedH5(
            base_dataset=dataset, path_to_cache_root=path, dataset_name='dataset-test', dataset_version='1.0', mode='a'
        )

        dataset = DatasetPairedList(dataset, [[0, 0]], pairing_sampler=PairingSamplerEnumerate())
        batch = dataset[0]

        # the sampler MUST be linked in paired dataset!
        assert batch['sampling_indices_min_zyx'][0] == batch['sampling_indices_min_zyx'][1]
        assert batch['sampling_indices_max_zyx'][0] == batch['sampling_indices_max_zyx'][1]


def test_paired_list_h5_sampler_blocks():
    with tempfile.TemporaryDirectory() as path:
        dataset = DatasetPath([os.path.join(here, 'resource/dicom_01')] * 2)
        dataset = DatasetImageReader(dataset, path_reader=partial(path_reader_dicom, image_namer=lambda h: h['Modality']))
        dataset = DatasetCachedH5(
            base_dataset=dataset,
            path_to_cache_root=path,
            dataset_name='dataset-test',
            dataset_version='1.0',
            mode='a',
            sampler=SamplerH5(CoordinateSamplerBlock(block_shape=(5, 32, 32))),
        )

        dataset = DatasetPairedList(dataset, [[0, 0]], pairing_sampler=PairingSamplerEnumerate())
        batch = dataset[0]

        assert np.abs(batch['MR_voxels'][0] - batch['MR_voxels'][1]).max() < 1e-6

        # the sampler MUST be linked in paired dataset!
        assert batch['sampling_indices_min_zyx'][0] == batch['sampling_indices_min_zyx'][1]
        assert batch['sampling_indices_max_zyx'][0] == batch['sampling_indices_max_zyx'][1]
