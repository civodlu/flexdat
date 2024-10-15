from typing import Dict, Optional, Tuple

import numpy as np

from .dataset import CoreDataset
from .dataset_dicom import VolumeExtractor, extract_itk_image_from_batch
from .itk import get_itk_size_mm_xyz
from .types import Batch


class DatasetResampleTargetCalculator(CoreDataset):
    """
    Calculate the targets resolution, origin, spacing, direction desired
    to achieve a given spacing or resolution.

    Designed to be used with `DatasetResample`.

    Example to resample data to a fixed shape:

    >>> dataset = DatasetPath([os.path.join(here, 'resource/dicom_01')])
    >>> dataset = DatasetImageReader(dataset, path_reader=partial(path_reader_dicom, image_namer=lambda h: h['Modality']))
    >>> dataset = DatasetResampleTargetCalculator(dataset, target_size_xyz=(16, 17, 18), volume_reference_name='')
    >>> dataset = DatasetResample(dataset, volume_names=('',))
    >>> batch = dataset[0]
    >>> batch['voxels'].shape
    (18, 17, 16)
    """

    def __init__(
        self,
        base_dataset: CoreDataset,
        volume_reference_name: str,
        target_spacing_xyz: Optional[Tuple[Optional[float], Optional[float], Optional[float]]] = None,
        target_size_xyz: Optional[Tuple[Optional[float], Optional[float], Optional[float]]] = None,
        volume_extractor: VolumeExtractor = extract_itk_image_from_batch,
        target_shape_name: str = 'target_shape',
        target_origin_name: str = 'target_origin',
        target_spacing_name: str = 'target_spacing',
        target_direction_name: str = 'target_direction',
    ):
        super().__init__()
        self.base_dataset = base_dataset
        self.volume_reference_name = volume_reference_name
        assert isinstance(volume_reference_name, str)
        self.target_spacing_xyz = target_spacing_xyz
        self.target_size_xyz = target_size_xyz
        self.volume_extractor = volume_extractor

        self.target_shape_name = target_shape_name
        self.target_origin_name = target_origin_name
        self.target_spacing_name = target_spacing_name
        self.target_direction_name = target_direction_name

        opt = int(target_spacing_xyz is not None) + int(target_size_xyz is not None)
        assert opt == 1, '`target_spacing_xyz` or `target_size_xyz` must be specified!'

    def __getitem__(self, index: int, context: Optional[Dict] = None) -> Optional[Batch]:
        batch = self.base_dataset.__getitem__(index, context)
        if batch is None:
            return None

        image = self.volume_extractor(batch, self.volume_reference_name)
        assert len(image.GetSize()) == 3, 'Must be a single image!'

        v_size_xyz = np.asarray(image.GetSize())

        if self.target_size_xyz is not None:
            target_size_xyz = np.asarray(
                [v if v is not None else v_size_xyz[v_n] for v_n, v in enumerate(self.target_size_xyz)]
            )
            target_spacing_xyz = get_itk_size_mm_xyz(image) / target_size_xyz

            batch[self.target_shape_name] = target_size_xyz[::-1]  # size = XYZ (ITK), shape = ZYX (numpy)
            batch[self.target_spacing_name] = target_spacing_xyz
            batch[self.target_origin_name] = tuple(image.GetOrigin())
            batch[self.target_direction_name] = np.asarray(image.GetDirection()).reshape((3, 3))

        else:
            target_spacing_xyz = np.asarray(
                [v if v is not None else v_size_xyz[v_n] for v_n, v in enumerate(self.target_spacing_xyz)]  # type: ignore
            )
            target_size_xyz = get_itk_size_mm_xyz(image) / target_spacing_xyz

            batch[self.target_shape_name] = target_size_xyz[::-1]  # size = XYZ (ITK), shape = ZYX (numpy)
            batch[self.target_spacing_name] = target_spacing_xyz
            batch[self.target_origin_name] = tuple(image.GetOrigin())
            batch[self.target_direction_name] = np.asarray(image.GetDirection()).reshape((3, 3))

        return batch

    def __len__(self) -> int:
        return len(self.base_dataset)
