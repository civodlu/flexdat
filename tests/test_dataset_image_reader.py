import os

from flexdat import DatasetImageReader, DatasetPath

here = os.path.abspath(os.path.dirname(__file__))


def test_ant_alignment():
    dataset = DatasetPath([os.path.join(here, 'resource/nifti_01/mr.nii.gz')])
    dataset = DatasetImageReader(dataset)

    b = dataset[0]
    assert b['image_voxels'].shape == (180, 258, 226)
