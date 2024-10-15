import os
from functools import partial

import numpy as np

from flexdat import DatasetPath, DatasetResample, DatasetResampleTargetCalculator
from flexdat.dataset_dicom import extract_itk_image_from_batch, path_reader_dicom
from flexdat.dataset_image_reader import DatasetImageReader
from flexdat.itk import get_itk_size_mm_xyz

here = os.path.abspath(os.path.dirname(__file__))


def test_resample_ct():
    dataset = DatasetPath(
        [os.path.join(here, 'resource/dicom_01')],
        target_shape=[(10, 20, 40)],
        target_spacing=[(1.0, 2.0, 3.0)],
        target_origin=[(-19.63, -39.58, 1.47)],
        target_direction=[np.eye(3)],
    )
    dataset = DatasetImageReader(dataset, path_reader=partial(path_reader_dicom, image_namer=lambda h: ''))
    dataset = DatasetResample(dataset, volume_names=('',))

    batch = dataset[0]
    assert batch['voxels'].shape == (10, 20, 40)
    assert np.abs(np.asarray(batch['origin']) - np.asarray([-19.63, -39.58, 1.47])).max() < 1e-5
    assert np.abs(np.asarray(batch['spacing']) - np.asarray([1.0, 2.0, 3.0])).max() < 1e-5


def test_resample_ct_optional_dont_exist():
    dataset = DatasetPath(
        [os.path.join(here, 'resource/dicom_01')],
        target_shape=[(10, 20, 40)],
        target_spacing=[(1.0, 2.0, 3.0)],
        target_origin=[(-19.63, -39.58, 1.47)],
        target_direction=[np.eye(3)],
    )
    dataset = DatasetImageReader(dataset, path_reader=partial(path_reader_dicom, image_namer=lambda h: ''))
    dataset = DatasetResample(dataset, volume_names=('',), optional_volume_names=('IMAGINARY'))

    batch = dataset[0]
    assert batch['voxels'].shape == (10, 20, 40)
    assert np.abs(np.asarray(batch['origin']) - np.asarray([-19.63, -39.58, 1.47])).max() < 1e-5
    assert np.abs(np.asarray(batch['spacing']) - np.asarray([1.0, 2.0, 3.0])).max() < 1e-5


def test_resample_ct_optional_exist():
    dataset = DatasetPath(
        [os.path.join(here, 'resource/dicom_01')],
        target_shape=[(10, 20, 40)],
        target_spacing=[(1.0, 2.0, 3.0)],
        target_origin=[(-19.63, -39.58, 1.47)],
        target_direction=[np.eye(3)],
    )
    dataset = DatasetImageReader(dataset, path_reader=partial(path_reader_dicom, image_namer=lambda h: ''))
    dataset = DatasetResample(dataset, volume_names=(), optional_volume_names=(''))

    batch = dataset[0]
    assert batch['voxels'].shape == (10, 20, 40)
    assert np.abs(np.asarray(batch['origin']) - np.asarray([-19.63, -39.58, 1.47])).max() < 1e-5
    assert np.abs(np.asarray(batch['spacing']) - np.asarray([1.0, 2.0, 3.0])).max() < 1e-5


def test_resample_specified_size():
    # keep everything the same EXCEPT the resolution
    dataset = DatasetPath([os.path.join(here, 'resource/dicom_01')])
    dataset_orig = DatasetImageReader(dataset, path_reader=partial(path_reader_dicom, image_namer=lambda h: ''))

    batch_orig = dataset_orig[0]
    v_orig = extract_itk_image_from_batch(batch_orig, '')
    v_orig_size_mm = get_itk_size_mm_xyz(v_orig)

    dataset = DatasetResampleTargetCalculator(dataset_orig, target_size_xyz=(11, 21, 31), volume_reference_name='')
    dataset = DatasetResample(dataset, volume_names=('',))

    batch = dataset[0]
    v_itk = extract_itk_image_from_batch(batch, '')
    v_itk_size_mm = get_itk_size_mm_xyz(v_itk)
    assert v_itk.GetSize() == (11, 21, 31)
    assert np.abs(np.asarray(v_orig.GetOrigin()) - np.asarray(v_itk.GetOrigin())).max() < 1e-5
    assert np.abs(v_itk_size_mm - v_orig_size_mm).max() < 1e-3


def test_resample_target():
    dataset = DatasetPath([os.path.join(here, 'resource/dicom_01')])

    dataset = DatasetImageReader(dataset, path_reader=partial(path_reader_dicom, image_namer=lambda h: ''))
    dataset = DatasetResampleTargetCalculator(dataset, target_size_xyz=(16, 17, 18), volume_reference_name='')
    dataset = DatasetResample(dataset, volume_names=('',))

    b = dataset[0]
    assert b['voxels'].shape == (18, 17, 16)  # XYZ to ZYX in ITK


def test_resample_specified_spacing():
    # keep everything the same EXCEPT the resolution
    dataset = DatasetPath([os.path.join(here, 'resource/dicom_01')])
    dataset_orig = DatasetImageReader(dataset, path_reader=partial(path_reader_dicom, image_namer=lambda h: ''))

    batch_orig = dataset_orig[0]
    v_orig = extract_itk_image_from_batch(batch_orig, '')
    v_orig_size_mm = get_itk_size_mm_xyz(v_orig)

    dataset = DatasetResampleTargetCalculator(dataset_orig, target_spacing_xyz=(1.0, 1.5, 2.0), volume_reference_name='')
    dataset = DatasetResample(dataset, volume_names=('',))

    batch = dataset[0]
    v_itk = extract_itk_image_from_batch(batch, '')
    v_itk_size_mm = get_itk_size_mm_xyz(v_itk)
    assert v_itk.GetSpacing() == (1.0, 1.5, 2.0)
    assert np.abs(np.asarray(v_orig.GetOrigin()) - np.asarray(v_itk.GetOrigin())).max() < 1e-5
    assert np.abs(v_itk_size_mm - v_orig_size_mm).max() < 1e-3

    dataset = DatasetPath([os.path.join(here, 'resource/dicom_01')])
    dataset = DatasetImageReader(dataset, path_reader=partial(path_reader_dicom, image_namer=lambda h: ''))
    dataset = DatasetResampleTargetCalculator(dataset, target_spacing_xyz=(1.0, 1.5, 2.0), volume_reference_name='')
    dataset = DatasetResample(dataset, volume_names=('',))

    batch = dataset[0]
    assert np.abs(np.asarray(batch['spacing']) - np.asarray([1.0, 1.5, 2.0])).max() < 1e-5
