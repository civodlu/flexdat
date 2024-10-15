import os

from flexdat import DatasetImageReader, DatasetPath
from flexdat.dataset_image_processing import (
    ImageProcessingCombine,
    post_processor_resample_fixed_spacing_images,
)

here = os.path.abspath(os.path.dirname(__file__))


def test_read_dict_images():
    dataset = DatasetPath(
        [
            {
                'mr': os.path.join(here, 'resource/nifti_01/mr.nii.gz'),
                'ct': os.path.join(here, 'resource/nifti_01/ct.nii.gz'),
            }
        ]
    )

    pp = ImageProcessingCombine(
        [
            post_processor_resample_fixed_spacing_images,
        ]
    )

    dataset = DatasetImageReader(dataset, image_postprocessing=pp)

    b = dataset[0]
    assert b['mr_voxels'].shape == (90, 129, 113)
    assert tuple(b['mr_spacing']) == (2.0, 2.0, 2.0)
    assert b['ct_voxels'].shape == (90, 129, 113)
    assert tuple(b['ct_spacing']) == (2.0, 2.0, 2.0)
