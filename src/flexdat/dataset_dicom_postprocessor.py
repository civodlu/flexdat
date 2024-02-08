from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union

import SimpleITK as sitk

from .dicom_folder import MetadataAdaptor
from .itk import (
    ItkInterpolatorType,
    SizeOptionalType,
    SpacingType,
    get_itk_interpolator,
    resample_spacing,
    resample_voxels,
    standard_orientation,
)

Header = Union[MetadataAdaptor, Dict[str, Any]]


def get_background_value_from_modality(modality: Literal['CT', 'MR', 'PT']) -> float:
    background_value = 0
    if modality == 'CT':
        background_value = -1024
    return background_value


def _resample(
    i: sitk.Image,
    h: Header,
    target_spacing_xyz: SpacingType,
    interpolator: ItkInterpolatorType,
) -> sitk.Image:
    background_value = get_background_value_from_modality(h['Modality'])
    i_p = resample_spacing(
        i, target_spacing_xyz=target_spacing_xyz, background_value=background_value, interpolator=interpolator
    )
    return i_p


def _resample_fixed_voxels(
    i: sitk.Image,
    h: Header,
    target_voxel_xyz: SizeOptionalType,
    interpolator: ItkInterpolatorType,
) -> sitk.Image:
    background_value = get_background_value_from_modality(h['Modality'])
    i_p = resample_voxels(i, target_voxel_xyz=target_voxel_xyz, background_value=background_value, interpolator=interpolator)
    return i_p


def post_processor_resample_fixed_voxels(
    images: Sequence[sitk.Image],
    headers: Sequence[Header],
    interpolator: ItkInterpolatorType = 'spline',
    target_voxel_xyz: SizeOptionalType = (256, 256, None),
) -> Tuple[Sequence[sitk.Image], Sequence[Header]]:
    """
    Resample the data to a given fixed number of voxels

    if an axis has `None` value in `target_voxel_xyz`, the number of voxels for this axis will not be changed.
    """
    images_processed = []
    for i, h in zip(images, headers):
        images_processed.append(_resample_fixed_voxels(i, h, target_voxel_xyz, interpolator=interpolator))
    return images_processed, headers


def post_processor_resample_fixed_spacing(
    images: Sequence[sitk.Image],
    headers: Sequence[Header],
    interpolator: ItkInterpolatorType = 'spline',
    target_spacing_xyz: SpacingType = (2.0, 2.0, 2.0),
) -> Tuple[Sequence[sitk.Image], Sequence[Header]]:
    """
    Resample the data to a given fixed spacing
    """
    images_processed = []
    for i, h in zip(images, headers):
        images_processed.append(_resample(i, h, target_spacing_xyz, interpolator=interpolator))
    return images_processed, headers


def post_processor_resample_fixed_reference(
    images: Sequence[sitk.Image],
    headers: Sequence[Header],
    geometry_reference_modality: str,
    interpolator: ItkInterpolatorType = 'spline',
    target_spacing_xyz: Optional[SpacingType] = (2.0, 2.0, 2.0),
) -> Tuple[Sequence[sitk.Image], Sequence[Header]]:
    """
    Resample the data to a given fixed spacing and a reference modality (e.g., resample at a given PET geometry)

    Args:
        geometry_reference_modality: the volume with this modality will be used as geometry reference for the other volumes
            reference geometry may be optionally resampled with a given `target_spacing_xyz`
    """
    ref_n = None
    for header_n, h in enumerate(headers):
        h_filename = getattr(h, 'filename')
        if h['Modality'] == geometry_reference_modality:
            assert (
                ref_n is None
            ), f'multiple identical modalities for the same patient! cannot select the proper one! {h_filename}'
            ref_n = header_n
    assert ref_n is not None, f'Base modality not found! {h_filename}'

    reference = images[ref_n]
    if target_spacing_xyz is not None and target_spacing_xyz != reference.GetSpacing():
        reference = _resample(reference, headers[ref_n], target_spacing_xyz, interpolator=interpolator)

    interpolator_itk = get_itk_interpolator(interpolator)
    images_processed = []
    for i_n, (i, h) in enumerate(zip(images, headers)):
        if i_n == ref_n:
            images_processed.append(reference)
        else:
            background_value = get_background_value_from_modality(h['Modality'])
            image_resampled = sitk.Resample(
                i,
                size=reference.GetSize(),
                interpolator=interpolator_itk,
                outputOrigin=reference.GetOrigin(),
                outputSpacing=reference.GetSpacing(),
                outputDirection=reference.GetDirection(),
                defaultPixelValue=background_value,
            )
            images_processed.append(image_resampled)

    return images_processed, headers


def post_processor_standard_axes(
    images: Sequence[sitk.Image],
    headers: Sequence[Header],
) -> Tuple[Sequence[sitk.Image], Sequence[Header]]:
    """
    Reorient the data with standard axes {(1, 0, 0), (0, 1, 0), (0, 0, 1)}
    """
    images_processed = [standard_orientation(v) for v in images]
    return images_processed, headers
