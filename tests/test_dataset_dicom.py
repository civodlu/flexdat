import os
from functools import partial

import numpy as np

from flexdat import DatasetMultipleDicoms, DatasetPath, DatasetSingleDicom
from flexdat.dataset_dicom import extract_itk_image_from_batch
from flexdat.dataset_dicom_postprocessor import (
    post_processor_resample_fixed_reference,
    post_processor_resample_fixed_spacing,
    post_processor_resample_fixed_voxels,
)
from flexdat.dicom_folder import read_dicom_folder

here = os.path.abspath(os.path.dirname(__file__))


def test_folder_multiple_series():
    path = os.path.join(here, 'resource/dicom_02')
    images, metadatas = read_dicom_folder(path)
    assert len(images) == 3
    assert len(metadatas) == 3

    ids = np.argsort([m['Modality'] for m in metadatas])

    assert metadatas[ids[0]]['Modality'] == 'CT'
    assert metadatas[ids[1]]['Modality'] == 'PT'
    assert metadatas[ids[2]]['Modality'] == 'SEG'

    assert images[ids[0]].GetSize() == (512, 512, 19)
    assert images[ids[1]].GetSize() == (400, 400, 20)
    assert images[ids[2]].GetSize() == (400, 400, 312)


def test_read_multiple_dicoms():
    path = os.path.join(here, 'resource/dicom_02')

    dataset = DatasetPath([path])
    dataset = DatasetMultipleDicoms(dataset)
    assert len(dataset) == 1

    batch = dataset[0]
    assert batch['SEG_voxels'].shape == (312, 400, 400)
    assert batch['CT_voxels'].shape == (19, 512, 512)
    assert batch['PT_voxels'].shape == (20, 400, 400)

    assert batch['SEG_SeriesInstanceUID'] == '1.3.6.1.4.1.14519.5.2.1.4219.6651.256283825970522789478274074883'
    assert batch['CT_SeriesInstanceUID'] == '1.3.6.1.4.1.14519.5.2.1.4219.6651.179400241151135528675710856782'
    assert batch['PT_SeriesInstanceUID'] == '1.3.6.1.4.1.14519.5.2.1.4219.6651.501625221769090036579553381234'


def test_read_single_dicoms():
    path = os.path.join(here, 'resource/dicom_02/image/PET/')

    dataset = DatasetPath([path])
    dataset = DatasetSingleDicom(dataset)
    assert len(dataset) == 1

    batch = dataset[0]
    assert batch['voxels'].shape == (20, 400, 400)
    assert batch['SeriesInstanceUID'] == '1.3.6.1.4.1.14519.5.2.1.4219.6651.501625221769090036579553381234'


def test_dataset_read_single_dicom_post_process_fixed_voxels():
    dataset = DatasetPath([os.path.join(here, 'resource/dicom_01')])
    dataset = DatasetSingleDicom(
        dataset,
        name_prefix='MR_T1_',
        dicoms_postprocessing=partial(post_processor_resample_fixed_voxels, target_voxel_xyz=(32, 33, None)),
    )

    batch = dataset[0]
    assert batch['MR_T1_voxels'].shape == (60, 33, 32)  # ZYX. `Z` should be the original size

    mr_itk = extract_itk_image_from_batch(batch, 'MR_T1_')
    assert mr_itk.GetSize() == (32, 33, 60)


def test_dataset_read_single_dicom_post_process_fixed_spacing():
    dataset = DatasetPath([os.path.join(here, 'resource/dicom_01')])
    dataset = DatasetSingleDicom(
        dataset,
        name_prefix='MR_T1_',
        dicoms_postprocessing=partial(post_processor_resample_fixed_spacing, target_spacing_xyz=(None, 1.5, 1.5)),
    )

    batch = dataset[0]
    assert (
        np.abs(batch['MR_T1_spacing'] - np.asarray((0.9375, 1.5, 1.5))).max() < 1e-6
    )  # ZYX. `Z` should be the original size


def test_dataset_read_single_dicom_post_process_fixed_spacing_single():
    dataset = DatasetPath([os.path.join(here, 'resource/dicom_01')])
    dataset = DatasetSingleDicom(
        dataset,
        name_prefix='MR_T1_',
        dicoms_postprocessing=partial(post_processor_resample_fixed_spacing, target_spacing_xyz=3.0),
    )

    batch = dataset[0]
    assert np.abs(batch['MR_T1_spacing'] - np.asarray((3, 3, 3))).max() < 1e-6  # ZYX. `Z` should be the original size


def test_dataset_read_multiple_dicoms_post_process_fixed_ref():
    dataset = DatasetPath([os.path.join(here, 'resource/dicom_02')])
    dataset = DatasetMultipleDicoms(
        dataset, dicoms_postprocessing=partial(post_processor_resample_fixed_reference, geometry_reference_modality='PT')
    )

    batch = dataset[0]
    ct_voxels = batch['CT_voxels']
    ct_spacing = batch['CT_spacing']
    ct_origin = batch['CT_origin']
    ct_direction = batch['CT_direction']

    pt_voxels = batch['PT_voxels']
    pt_spacing = batch['PT_spacing']
    pt_origin = batch['PT_origin']
    pt_direction = batch['PT_direction']

    assert (pt_origin == ct_origin).all()
    assert (pt_spacing == ct_spacing).all()
    assert pt_voxels.shape == ct_voxels.shape
    assert (pt_direction == ct_direction).all()

    seg_voxels = batch['SEG_voxels']
    assert pt_voxels.shape == seg_voxels.shape
