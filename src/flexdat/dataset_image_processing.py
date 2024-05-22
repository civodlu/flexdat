import logging
import os
from typing import Callable, Dict, Sequence, Union

import SimpleITK as sitk

from .itk import ItkInterpolatorType, SpacingType, resample_spacing
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
) -> Dict[str, sitk.Image]:
    """
    Rename the volume by position in the sequence
    """
    renamed = {}
    for item_n, (name, value) in enumerate(images.items()):
        name = os.path.basename(name)
        name = name.replace('.nii.gz', '').replace('.nii', '') + '_'

        if len(images) > 1:
            name_str = fixed_name + str(item_n)
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
