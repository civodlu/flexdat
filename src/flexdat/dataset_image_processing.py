import logging
import os
from typing import Callable, Dict, Optional, Sequence, Union

import SimpleITK as sitk

from .itk import (
    ItkInterpolatorType,
    SpacingType,
    get_itk_interpolator,
    resample_spacing,
)
from .types import Batch

logger = logging.getLogger(__name__)


ImagePostprocessor = Callable[[Dict[str, sitk.Image], Batch], Dict[str, sitk.Image]]


class ImageProcessingCombine:
    """
    Combine multiple processors
    """

    def __init__(self, processors: Sequence[ImagePostprocessor]):
        self.processors = processors

    def __call__(self, images: Dict[str, sitk.Image], batch: Batch) -> Dict[str, sitk.Image]:
        for p in self.processors:
            images = p(images, batch)
        return images


def image_postprocessing_rename_fixed(
    images: Dict[str, sitk.Image],
    batch: Batch,
    fixed_name: str = '',
    postfix: str = '_',
) -> Dict[str, sitk.Image]:
    """
    Rename the volume by position in the sequence
    """
    renamed = {}
    for item_n, (name, value) in enumerate(images.items()):
        name = os.path.basename(name)
        name = name.replace('.nii.gz', '').replace('.nii', '') + '_'

        if len(images) > 1:
            name_str = fixed_name + str(item_n) + postfix
        else:
            name_str = fixed_name

        renamed[name_str] = value

    return renamed


def image_postprocessing_rename(
    images: Dict[str, sitk.Image],
    batch: Batch,
    name_fn: Callable[[str], str] = lambda name: name.replace('.nii.gz', '').replace('.nii', '') + '_',
) -> Dict[str, sitk.Image]:
    """
    Rename the volume by removing extension and root directory
    """
    renamed = {}
    for name, value in images.items():
        name = os.path.basename(name)
        name = name_fn(name)
        renamed[name] = value

    return renamed


def post_processor_resample_fixed_spacing_images(
    images: Dict[str, sitk.Image],
    batch: Batch,
    interpolators: Union[ItkInterpolatorType, Dict[str, ItkInterpolatorType]] = 'spline',
    background_values: Union[float, Dict[str, float]] = 0,
    target_spacing_xyz: SpacingType = (2.0, 2.0, 2.0),
) -> Dict[str, sitk.Image]:
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
    return images_processed


def post_processor_resample_reference_images(
    images: Dict[str, sitk.Image],
    batch: Batch,
    reference_name: str,
    interpolators: Union[ItkInterpolatorType, Dict[str, ItkInterpolatorType]] = 'spline',
    background_values: Optional[Union[float, Dict[str, float]]] = None,
    target_spacing_xyz: Optional[SpacingType] = (2.0, 2.0, 2.0),
):
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

    return resampled_images
