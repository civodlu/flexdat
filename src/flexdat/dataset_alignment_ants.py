from copy import copy
from typing import Callable, Dict, Optional, Sequence

import SimpleITK as sitk

from .dataset import CoreDataset
from .dataset_dicom import (
    VolumeExtractor,
    VolumeSerializer,
    extract_itk_image_from_batch,
    itk_serializer,
)
from .itk import make_sitk_like
from .types import Batch
from .utils import np_array_type_clip


class DatasetAlignmentANTs(CoreDataset):
    """
    Calculate the alignment between a `fixed` (i.e., reference) and a `moving` volume

    Example:

    >>> dataset = DatasetPath(['folder/fixed', 'folder/moving'])  # has `ct.nii.gz` in this folder
    >>> dataset = DatasetImageReader(dataset)
    >>> transform = TransformCompose([
            # use the `meaningful` part of the intensity range to perform alignment
            TransformNormalizeRange(
                voxels_name='fixed_ct_voxels',
                voxel_range_min=-300,
                voxel_range_max=300,
                output_range=(0, 1)),
            TransformNormalizeRange(
                voxels_name='moving_ct_voxels',
                voxel_range_min=-300,
                voxel_range_max=300,
                output_range=(0, 1)),
        ])
    >>> dataset = DatasetPaired(
            dataset,
            [[0, 1]],
            pairing_sampler=PairingSamplerRandom(pairing_key_prefix=('fixed_', 'moving_'))
        )
    >>> dataset = DatasetAlignmentANTs(
            dataset,
            fixed='fixed_ct_',
            moving='moving_ct_',
            pre_transform_alignment=transform,
            resample_prefix='aligned_'
        )
    >>> b = dataset[0]
    b['aligned_moving_ct_voxels'] holds the aligned volume to target.
    """

    def __init__(
        self,
        base_dataset: CoreDataset,
        fixed: str,
        moving: str,
        volume_extractor: VolumeExtractor = extract_itk_image_from_batch,
        volume_serializer: VolumeSerializer = itk_serializer,
        pre_transform_alignment: Optional[Callable[[Batch], Batch]] = None,
        transform: Optional[Callable[[Batch], Batch]] = None,
        alignment_kwargs: Dict = {'type_of_transform': 'SyN'},
        resample_volumes: Optional[Sequence[str]] = None,
        resample_volumes_segmentations: Optional[Sequence[str]] = None,
        resample_prefix: str = '',
        resample_kwargs: Dict = {'interpolator': 'bSpline'},
        resample_reference_volume_name: Optional[str] = None,
        background_values: Dict[str, float] = {},
        default_background_value: float = 0,
        record_tfm: bool = False,
    ) -> None:
        """
        Parameters:
            pre_transform_alignment: pre-transform applied on the volumes just
                before alignment (e.g., focus alignment on intensity range)
            resample_volumes: name of the volumes to be extracted using `volume_extractor`
                and resampled using the moving->fixed alignment. Finally, they are serialized
                with `volume_serializer` and `resample_prefix` used prefix
            resample_volumes_segmentations: the name of the segmentations to be resampled.
                This is handled separately to avoid changing segmentation IDs caused by interpolation
            background_values: the background value for each volume resampled. If not specified, `0` is used
            resample_reference_volume_name: the name of the volume to be used as the reference geometry for
                the resampling
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.fixed = fixed
        self.moving = moving
        self.volume_extractor = volume_extractor
        self.volume_serializer = volume_serializer
        self.pre_transform_alignment = pre_transform_alignment
        self.transform = transform
        self.alignment_kwargs = alignment_kwargs
        self.record_tfm = record_tfm
        if resample_volumes_segmentations is None:
            self.resample_volumes_segmentations: Sequence[str] = ()
        else:
            self.resample_volumes_segmentations = resample_volumes_segmentations

        if resample_volumes is None:
            resample_volumes = (moving,)
        self.resample_volumes = resample_volumes
        self.resample_kwargs = resample_kwargs
        self.resample_prefix = resample_prefix
        self.background_values = background_values
        self.default_background_value = default_background_value
        self.resample_reference_volume_name = resample_reference_volume_name

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int, context: Optional[Dict] = None) -> Optional[Batch]:
        import ants

        from .utils_ants import ants_to_itk_image, itk_to_ants_image

        batch_orig = self.base_dataset.__getitem__(index, context)
        if batch_orig is None:
            return None

        batch = copy(batch_orig)  # make sure we don't modify the original batch
        if self.pre_transform_alignment is not None:
            batch = self.pre_transform_alignment(batch)

        moving_itk = self.volume_extractor(batch, self.moving)
        moving_ants = itk_to_ants_image(moving_itk)
        fixed_itk = self.volume_extractor(batch, self.fixed)
        fixed_ants = itk_to_ants_image(fixed_itk)
        tfm = ants.registration(fixed=fixed_ants, moving=moving_ants, **self.alignment_kwargs)

        batch_result = copy(batch_orig)
        if self.record_tfm:
            # the transform may be used for other tasks
            # optionally return it with the batch
            assert len(tfm['fwdtransforms']) == 1
            tfm_itk = sitk.ReadTransform(tfm['fwdtransforms'][0])
            batch_result['transform_itk'] = tfm_itk

        resample_reference_ants = fixed_ants
        if self.resample_reference_volume_name is not None:
            resample_reference_itk = self.volume_extractor(batch, self.resample_reference_volume_name)
            resample_reference_ants = itk_to_ants_image(resample_reference_itk)

        for name in list(self.resample_volumes) + list(self.resample_volumes_segmentations):
            # apply specific interpolation & remove any preprocessing for the alignment
            moving_orig_itk = self.volume_extractor(batch_orig, name)
            moving_orig_ants = itk_to_ants_image(moving_orig_itk)

            background_value = self.background_values.get(name)
            if background_value is None:
                background_value = self.default_background_value

            if name in self.resample_volumes_segmentations:
                # segmentation: we don't want ANY interpolation
                moving_aligned_ants = ants.apply_transforms(
                    fixed=resample_reference_ants,
                    moving=moving_orig_ants,
                    transformlist=tfm['fwdtransforms'],
                    interpolator='nearestNeighbor',
                    defaultvalue=background_value,
                )
            else:
                moving_aligned_ants = ants.apply_transforms(
                    fixed=resample_reference_ants,
                    moving=moving_orig_ants,
                    transformlist=tfm['fwdtransforms'],
                    defaultvalue=background_value,
                    **self.resample_kwargs,
                )

            # unfortunately, ANTs will change the data types: convert back to original
            # this is not straightforward: e.g., if we ahave an unsigned type (e.g., segmentation)
            # the interpolation can be a problem as ants will cast to float and some voxels
            # may be negative.
            # BEWARE: do NOT use `sitk.Cast`!!!
            moving_aligned_itk = ants_to_itk_image(moving_aligned_ants)
            moving_aligned_np = sitk.GetArrayViewFromImage(moving_aligned_itk)
            moving_orig_np_dtype = sitk.GetArrayViewFromImage(moving_orig_itk).dtype
            moving_aligned_np = np_array_type_clip(moving_aligned_np, moving_orig_np_dtype)
            moving_aligned_itk = make_sitk_like(moving_aligned_np, moving_aligned_itk)

            moving_attributes = self.volume_serializer(
                volume=moving_aligned_itk,
                header={},
                index=0,
                base_name=name,
            )
            for name, value in moving_attributes.items():
                batch_result[self.resample_prefix + name] = value

            """
            from flexdat.itk import write_nifti
            write_nifti(moving_aligned_itk, '/tmp/experiments/output/moving_aligned_itk.nii.gz')
            ants.image_write(moving_aligned_ants, '/tmp/experiments/output/moving_aligned.nii.gz')
            ants.image_write(tfm['warpedmovout'], '/tmp/experiments/output/moving_aligned_output.nii.gz')
            ants.image_write(fixed_ants, '/tmp/experiments/output/fixed.nii.gz')
            ants.image_write(moving_ants, '/tmp/experiments/output/moving.nii.gz')
            """
        return batch_result
