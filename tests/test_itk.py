import numpy as np
import SimpleITK as sitk
import torch

import flexdat.itk


def test_make_itk():
    v = flexdat.itk.make_sitk_image(
        np.full([4, 5, 6], fill_value=42, dtype=int),
        origin_xyz=(-2, -3, -4),
        spacing_xyz=(1, 2, 3),
    )

    v_attributes = flexdat.itk.get_sitk_image_attributes(v)
    assert np.abs(np.asarray(v_attributes['origin']) - np.asarray([-2, -3, -4])).max() < 1e-5
    assert np.abs(np.asarray(v_attributes['spacing']) - np.asarray([1, 2, 3])).max() < 1e-5
    assert (np.asarray(v_attributes['shape']) == np.asarray([6, 5, 4])).all()

    v_np = sitk.GetArrayViewFromImage(v)
    assert (v_np == 42).all()

    v_zeros = flexdat.itk.make_sitk_zero_like(v)
    v_zeros_np = sitk.GetArrayViewFromImage(v_zeros)
    assert (v_zeros_np == 0).all()

    v_like = flexdat.itk.make_sitk_like(np.full([4, 5, 6], fill_value=2), target=v)
    v_like_attributes = flexdat.itk.get_sitk_image_attributes(v_zeros)
    for name, value in v_like_attributes.items():
        assert np.abs(np.asarray(value) - np.asarray(v_attributes[name])).max() < 1e-5
    v_like_np = sitk.GetArrayViewFromImage(v_like)
    assert (v_like_np == 2).all()

    # NP shape = reversed(ITK size)
    size_mm = tuple(flexdat.itk.get_itk_size_mm_xyz(v))
    assert size_mm == (6 * 1, 5 * 2, 4 * 3)

    voxel2world = flexdat.itk.get_voxel_to_mm_transform(v)
    assert (torch.eye(3) * torch.asarray([[1, 2, 3]]) - voxel2world[:3, :3]).abs().max() < 1e-5
    assert (voxel2world[:3, 3] - torch.asarray([-2, -3, -4])).abs().max() < 1e-5
    rotation4x4 = flexdat.itk.get_itk_rotation_4x4(v)
    assert (rotation4x4 - torch.eye(4)).abs().max() < 1e-5

    # the data is ALREADY in LPS format, expecting no changes!
    v_standard = flexdat.itk.standard_orientation(v)
    assert v_standard.GetDirection() == v.GetDirection()
    assert v_standard.GetOrigin() == v.GetOrigin()
    assert v_standard.GetSpacing() == v.GetSpacing()


def test_cropping():
    i = np.zeros([10, 12, 14], dtype=float)
    i[3:8, 5:8, 10:14] = np.random.randn(5, 3, 4) + 42.0

    v = flexdat.itk.make_sitk_image(
        i,
        origin_xyz=(-2, -3, -4),
        spacing_xyz=(1, 2, 3),
    )

    v_cropped = flexdat.itk.crop_image(v)
    v_cropped_np = sitk.GetArrayViewFromImage(v_cropped)
    assert v_cropped.GetSize() == (4, 3, 5)
    assert np.abs(v_cropped_np - i[3:8, 5:8, 10:14]).max() < 1e-6
    v_cropped_o = v_cropped.GetOrigin()
    v_cropped_o_expected = v.TransformContinuousIndexToPhysicalPoint((10.0, 5.0, 3.0))
    assert v_cropped_o == v_cropped_o_expected

    tfm_v_index_to_world = flexdat.itk.get_voxel_to_mm_transform(v)
    v_cropped_o_expected_manual = flexdat.itk.apply_homogeneous_affine_transform(tfm_v_index_to_world, (10, 5, 3)).numpy()
    assert (v_cropped_o == v_cropped_o_expected_manual).all()
