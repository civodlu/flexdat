from copy import copy
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Union

import numpy as np
import SimpleITK as sitk

from .dataset import CoreDataset
from .dataset_dicom import (
    VolumeExtractor,
    VolumeSerializer,
    extract_itk_image_from_batch,
    itk_serializer,
)
from .dataset_dicom_postprocessor import get_background_value_from_modality
from .itk import get_itk_interpolator
from .types import Batch


def default_background_value(batch: Batch, key_name: str, modality_name: Optional[str]) -> float:
    # if we have modality as tag, use this!
    if modality_name is not None:
        modality = batch.get(modality_name)
        if modality is not None:
            return get_background_value_from_modality(modality)

    # if not, try to guess from the name
    if 'ct_' in key_name or 'CT_' in key_name or '_ct' in key_name or '_CT' in key_name:
        # CT is the exception
        return -1024.0

    # default PET/MR
    return 0


class DatasetResample(CoreDataset):
    """
    Resample a volume to a given geometry.

    The geometry information must be contained in the batch.

    Beware <shape> (numpy ZYX) vs size (ITK XYZ).

    Example (read the target resampling from batch):

    >>> dataset = DatasetPath([
           os.path.join(here, 'resource/dicom_01')],
           target_shape=[(10, 20, 40)],
           target_spacing=[(1.0, 2.0, 3.0)],
           target_origin=[(-19.63, -39.58, 1.47)],
           target_direction=[np.eye(3)])
    >>> dataset = DatasetSingleDicom(dataset, name_prefix='')
    >>> dataset = DatasetResample(dataset, volume_names=('',))


    Example (calculate from batch / specified)

    >>> dataset = DatasetPath([os.path.join(here, 'resource/dicom_01')])
    >>> dataset = DatasetSingleDicom(dataset, name_prefix='')
    >>> dataset = DatasetResampleTargetCalculator(dataset, target_spacing_xyz=(1.0, 1.5, 2.0), volume_reference_name='')
    >>> dataset = DatasetResample(dataset, volume_names=('',))
    """

    def __init__(
        self,
        base_dataset: CoreDataset,
        volume_names: Union[Sequence[str], str],
        optional_volume_names: Union[Sequence[str], str] = (),
        transform: Optional[Callable[[Batch], Batch]] = None,
        pre_transform: Optional[Callable[[Batch], Batch]] = None,
        volume_extractor: VolumeExtractor = extract_itk_image_from_batch,
        volume_serializer: VolumeSerializer = itk_serializer,
        target_origin_name: Optional[str] = 'target_origin',
        target_spacing_name: Optional[str] = 'target_spacing',
        target_direction_name: Optional[str] = 'target_direction',
        target_shape_name: Optional[str] = 'target_shape',
        modality_name: Optional[str] = 'Modality',
        background_value: Callable[[Batch, str, Optional[str]], float] = default_background_value,
        interpolator_name: Literal['spline', 'linear', 'nearest'] = 'spline',
        segmentation_dtype: Sequence[Any] = (np.uint8,),
    ):
        """
        Parameters:
            target_shape_name: the name of the shape of the volume in the batch. Specified as ZYX order
            target_origin_name: the name of the origin of the target volume. Specified in XYZ
            target_spacing_name: the name of the spacing of the target volume. Specified in XYZ
            target_direction_name: the name of the Voxel->patient space orientation as (3,3) rotation matrix in XYZ order
        """
        super().__init__()
        self.base_dataset = base_dataset
        if isinstance(volume_names, str):
            volume_names = (volume_names,)
        self.volume_names = volume_names
        if isinstance(optional_volume_names, str):
            optional_volume_names = (optional_volume_names,)
        self.optional_volume_names = optional_volume_names

        self.transform = transform
        self.volume_extractor = volume_extractor
        self.volume_serializer = volume_serializer
        self.interpolator_itk = get_itk_interpolator(interpolator_name)

        self.target_origin_name = target_origin_name
        self.target_spacing_name = target_spacing_name
        self.target_direction_name = target_direction_name
        self.target_shape_name = target_shape_name
        self.background_value = background_value
        self.modality_name = modality_name
        self.segmentation_dtype = segmentation_dtype
        self.pre_transform = pre_transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int, context: Optional[Dict] = None) -> Optional[Batch]:
        batch = self.base_dataset.__getitem__(index, context)
        if batch is None:
            return None

        batch = copy(batch)  # make sure we don't modify the original batch
        if self.pre_transform is not None:
            batch = self.pre_transform(batch)

        target_spacing = None
        if self.target_spacing_name is not None:
            target_spacing = batch[self.target_spacing_name]
            target_spacing = [float(v) for v in target_spacing]
            assert len(target_spacing) == 3

        target_origin = None
        if self.target_origin_name is not None:
            target_origin = batch[self.target_origin_name]
            target_origin = [float(v) for v in target_origin]
            assert len(target_origin) == 3

        target_direction = None
        if self.target_direction_name is not None:
            target_direction = batch[self.target_direction_name]
            assert target_direction.shape == (3, 3)
            target_direction = [float(v) for v in target_direction.ravel()]

        target_shape_xyz = None
        if self.target_shape_name is not None:
            target_shape_xyz = batch[self.target_shape_name][::-1]  # ZYX to XYZ shape
            target_shape_xyz = [int(v) for v in target_shape_xyz]
            assert len(target_shape_xyz) == 3

        def apply_resampling(name: str, optional: bool) -> None:
            try:
                volume_itk = self.volume_extractor(batch, name)
            except Exception as e:
                if not optional:
                    raise e
                return  # skip!

            # if None, copy attributes from the original volume
            nonlocal target_origin
            nonlocal target_spacing
            nonlocal target_direction
            nonlocal target_shape_xyz
            if target_origin is None:
                target_origin = volume_itk.GetOrigin()
            if target_spacing is None:
                target_spacing = volume_itk.GetSpacing()
            if target_direction is None:
                target_direction = volume_itk.GetDirection()
            if target_shape_xyz is None:
                target_shape_xyz = volume_itk.GetSize()

            # check if the target geometry is different from current geometry to avoid unnecessary
            # resampling
            origin_diff_max = np.abs(np.asarray(volume_itk.GetOrigin()) - np.asarray(target_origin)).max()
            spacing_diff_max = np.abs(np.asarray(volume_itk.GetSpacing()) - np.asarray(target_spacing)).max()
            dir_diff_max = np.abs(np.asarray(volume_itk.GetDirection()) - np.asarray(target_direction)).max()
            shape_max = np.abs(np.asarray(volume_itk.GetSize()) - np.asarray(target_shape_xyz)).max()
            tol = 1e-4
            if shape_max < tol and dir_diff_max < tol and spacing_diff_max < tol and origin_diff_max < tol:
                # the geometry is IDENTICAL. Do not Resample!
                # we avoid resampling as much as possible to avoid resampling artifacts
                # (e.g., values < 0 with spline interpolator)
                return

            # there is a difference in the geometry, resample the volume
            interpolator_itk = self.interpolator_itk
            volume_dtype = sitk.GetArrayViewFromImage(volume_itk).dtype
            if volume_dtype in self.segmentation_dtype:
                # spacial case: if we have segmentations, do NOT interpolate!
                interpolator_itk = sitk.sitkNearestNeighbor

            background_value = self.background_value(batch, name, self.modality_name)
            volume_itk_resampled = sitk.Resample(
                volume_itk,
                size=target_shape_xyz,
                # for segmentation, we MUST use nearest neighbor (we want integer values only)
                interpolator=interpolator_itk,
                outputOrigin=target_origin,
                outputSpacing=target_spacing,
                outputDirection=target_direction,
                defaultPixelValue=background_value,
            )

            volume_itk_resampled_serialized = self.volume_serializer(
                volume=volume_itk_resampled,
                header=batch,
                index=0,
                base_name=name,
            )
            for name, value in volume_itk_resampled_serialized.items():
                batch[name] = value

        for name in self.volume_names:
            apply_resampling(name, optional=False)
        for name in self.optional_volume_names:
            apply_resampling(name, optional=True)
        return batch
