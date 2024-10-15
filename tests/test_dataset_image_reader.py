import os

import numpy as np

from flexdat import DatasetImageReader, DatasetPath
from flexdat.dataset_image_processing import (
    ImageProcessingCombine,
    image_postprocessing_rename,
    post_processor_resample_fixed_spacing_images,
)

here = os.path.abspath(os.path.dirname(__file__))


def test_read_single_image():
    dataset = DatasetPath([os.path.join(here, 'resource/nifti_01/mr.nii.gz')])
    dataset = DatasetImageReader(dataset)

    b = dataset[0]
    assert b['mr_voxels'].shape == (180, 258, 226)


def test_read_multiple_images():
    dataset = DatasetPath(
        [(os.path.join(here, 'resource/nifti_01/mr.nii.gz'), os.path.join(here, 'resource/nifti_01/ct.nii.gz'))]
    )

    pp = ImageProcessingCombine(
        [
            image_postprocessing_rename,
            post_processor_resample_fixed_spacing_images,
        ]
    )

    dataset = DatasetImageReader(dataset, image_postprocessing=pp)

    b = dataset[0]
    assert b['mr_voxels'].shape == (90, 129, 113)
    assert tuple(b['mr_spacing']) == (2.0, 2.0, 2.0)
    assert b['ct_voxels'].shape == (90, 129, 113)
    assert tuple(b['ct_spacing']) == (2.0, 2.0, 2.0)


def test_image_dataset_read_folder():
    here = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(here, 'resource/nifti_01')
    dataset = DatasetPath([path])
    dataset = DatasetImageReader(dataset)

    batch = dataset[0]
    assert 'ct_voxels' in batch
    assert 'mask_voxels' in batch
    assert 'mr_voxels' in batch
    assert batch['ct_voxels'].shape == (180, 258, 226)
    assert np.abs(batch['ct_spacing'] - np.asarray((1.0, 1.0, 1.0))).max() < 1e-5
