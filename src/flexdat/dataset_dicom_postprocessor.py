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
