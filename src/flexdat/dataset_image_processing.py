import logging
import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import SimpleITK as sitk

from .itk import (
    ItkInterpolatorType,
    SpacingType,
    get_itk_interpolator,
    resample_spacing,
)
from .types import Batch

logger = logging.getLogger(__name__)


# <Image, possibly Headers>
ProcessingOutput = Tuple[Dict[str, sitk.Image], Optional[Dict[str, Any]]]

# Input = (Images, Batch, Headers) Return=(Images, Headers)
ImagePostprocessor = Callable[[Dict[str, sitk.Image], Batch, Optional[Dict[str, Any]]], ProcessingOutput]


class ImageProcessingCombine:
    """
    Combine multiple processors
    """

    def __init__(self, processors: Sequence[ImagePostprocessor]):
        self.processors = processors

    def __call__(self, images: Dict[str, sitk.Image], batch: Batch, headers: Optional[Dict[str, Any]]) -> ProcessingOutput:
        for p in self.processors:
            images, headers = p(images, batch, headers)
        return images, headers


def image_postprocessing_rename_fixed(
    images: Dict[str, sitk.Image],
    batch: Batch,
    headers: Optional[Dict[str, Any]],
    fixed_name: str = '',
    postfix: str = '_',
) -> ProcessingOutput:
    """
    Rename the volume by position in the sequence
    """
    images_renamed = {}
    headers_renamed = {}
    for item_n, (name, value) in enumerate(images.items()):
        name = os.path.basename(name)
        name = name.replace('.nii.gz', '').replace('.nii', '').replace('.mha', '') + '_'

        if len(images) > 1:
            name_str = fixed_name + str(item_n) + postfix
        else:
            name_str = fixed_name

        images_renamed[name_str] = value

        if headers is not None:
            value_header = headers.get(name)
            if value_header is not None:
                headers_renamed[name_str] = value_header

    return images_renamed, headers_renamed


def rename_remove_extension(name: str, postfix: str = '_') -> str:
    if len(name) == 0:
        # if empty, don't add the postfix
        return ''

    name_renamed = name.replace('.nii.gz', '').replace('.nii', '').replace('.mha', '')
    if not name.endswith(postfix):
        # don't add 2 postfixes if they are the same
        name_renamed = name_renamed + postfix

    return name_renamed


def image_postprocessing_rename(
    images: Dict[str, sitk.Image],
    batch: Batch,
    headers: Optional[Dict[str, Any]],
    name_fn: Callable[[str], str] = rename_remove_extension,
) -> ProcessingOutput:
    """
    Rename the volume by removing extension and root directory
    """
    images_renamed = {}
    headers_renamed = {}
    for name, value in images.items():
        renamed = os.path.basename(name)
        renamed = name_fn(renamed)
        images_renamed[renamed] = value

        if headers is not None and name in headers:
            headers_renamed[renamed] = headers[name]

    return images_renamed, headers_renamed


def post_processor_resample_fixed_spacing_images(
    images: Dict[str, sitk.Image],
    batch: Batch,
    headers: Optional[Dict[str, Any]],
    interpolators: Union[ItkInterpolatorType, Dict[str, ItkInterpolatorType]] = 'spline',
    background_values: Union[float, Dict[str, float]] = 0,
    target_spacing_xyz: SpacingType = (2.0, 2.0, 2.0),
) -> ProcessingOutput:
    """
    Resample the data to a given fixed spacing
    """
    images_processed = {}
    for name, image in images.items():
        if isinstance(interpolators, Dict):
            interpolator: ItkInterpolatorType = interpolators.get(name)  # type: ignore
        else:
            interpolator: ItkInterpolatorType = interpolators  # type: ignore

        if isinstance(background_values, Dict):
            background_value: float = background_values.get(name)  # type: ignore
        else:
            background_value: float = background_values  # type: ignore

        resampled = resample_spacing(
            image,
            target_spacing_xyz=target_spacing_xyz,
            interpolator=interpolator,
            background_value=background_value,
        )
        images_processed[name] = resampled
    return images_processed, headers


def post_processor_resample_reference_images(
    images: Dict[str, sitk.Image],
    batch: Batch,
    headers: Optional[Dict[str, Any]],
    reference_name: str,
    interpolators: Union[ItkInterpolatorType, Dict[str, ItkInterpolatorType]] = 'spline',
    background_values: Optional[Union[float, Dict[str, float]]] = None,
    target_spacing_xyz: Optional[SpacingType] = (2.0, 2.0, 2.0),
) -> ProcessingOutput:
    """
    Resample the data to a given geometry. Optionally, set a fixed spacing
    """
    image_ref = images[reference_name]

    # make interpolator/background specific to each volume
    if not isinstance(interpolators, dict):
        interpolators = {name: interpolators for name in images.keys()}
    if background_values is None:
        background_values = {name: float(sitk.GetArrayViewFromImage(v).min()) for name, v in images.items()}
    elif not isinstance(interpolators, dict):
        background_values = {name: background_values for name in images.keys()}
    assert background_values is not None

    # first resample the reference volume
    if target_spacing_xyz is not None:
        image_ref = resample_spacing(
            image_ref,
            target_spacing_xyz=target_spacing_xyz,
            segmentation_dtype=None,  # do not rely on type
            interpolator=interpolators[reference_name],
        )

    # then resamples all the other volumes to reference
    resampled_images = {reference_name: image_ref}
    for name, image in images.items():
        if name != reference_name:
            image_r = sitk.Resample(
                image,
                size=image_ref.GetSize(),  # type: ignore
                interpolator=get_itk_interpolator(interpolators[name]),
                outputOrigin=image_ref.GetOrigin(),
                outputSpacing=image_ref.GetSpacing(),
                outputDirection=image_ref.GetDirection(),
                defaultPixelValue=background_values[name],
            )
            resampled_images[name] = image_r

    return resampled_images, headers
