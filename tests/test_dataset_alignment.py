import os

from flexdat import DatasetAlignmentANTs, DatasetMultipleDicoms, DatasetPath

here = os.path.abspath(os.path.dirname(__file__))


def test_ant_alignment():
    dataset = DatasetPath([os.path.join(here, 'resource/dicom_02/')])
    dataset = DatasetMultipleDicoms(dataset)
    dataset = DatasetAlignmentANTs(
        dataset,
        fixed='CT_',
        moving='PT_',
        resample_prefix='aligned_',
        resample_volumes=('PT_',),
        resample_volumes_segmentations=('SEG_',),
        alignment_kwargs={'type_of_transform': 'QuickRigid'},  # just have a quick alignment only to exercise the API
    )

    b = dataset[0]

    assert 'aligned_PT_voxels' in b
    assert 'aligned_SEG_voxels' in b
    assert 'PT_voxels' in b
    assert 'SEG_voxels' in b
    assert 'CT_voxels' in b
