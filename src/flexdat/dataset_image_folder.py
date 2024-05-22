import logging
import os
from glob import glob
from typing import Callable, Dict, Optional, Sequence

import SimpleITK as sitk

from .dataset import CoreDataset
from .dataset_dicom import VolumeSerializer, itk_serializer
from .dataset_image_processing import (
    ImagePostprocessor,
    ImageProcessingCombine,
    image_postprocessing_rename,
    image_postprocessing_rename_fixed,
)
from .itk import read_nifti
from .types import Batch

logger = logging.getLogger(__name__)


ImageLoader = Callable[[str], sitk.Image]


class DatasetImageFolder(CoreDataset):
    """
    Dataset pointing to a local path, containing multiple nifty images of the same patient
    (e.g., PET/CT modality or multiple MR sequences)

    `nifti_extensions` will be used to locate nifti files in the folder.

    Example:

    >>> paths = ['/path/1', '/path/2']
    >>> dataset = DatasetPath(paths)
    >>> dataset = DatasetImageFolder(dataset)
    >>> batch = dataset[0]
    """

    def __init__(
        self,
        base_dataset: CoreDataset,
        nifti_loader: ImageLoader = read_nifti,
        nifty_postprocessing: Optional[ImagePostprocessor] = image_postprocessing_rename,
        path_name: str = 'path',
        nifti_extensions: Sequence[str] = ('.nii.gz',),
        volume_serializer: VolumeSerializer = itk_serializer,
        transform: Optional[Callable[[Batch], Batch]] = None,
    ):
        """
        Args:
            base_dataset: the base dataset to load DICOM from
            dicom_loader: how to load the DICOM file/directory
            path_name: location name in the base dataset
            volume_serializer: extract information from the image (e.g., voxel, coordinate system)
        """
        super().__init__()
        self.nifti_loader = nifti_loader
        self.nifty_postprocessing = nifty_postprocessing
        self.path_name = path_name
        self.base_dataset = base_dataset
        self.volume_serializer = volume_serializer
        self.transform = transform
        self.nifti_extensions = nifti_extensions
        assert not isinstance(nifti_extensions, str), 'must a a sequence of string!'

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int, context: Optional[Dict] = None) -> Optional[Batch]:
        batch = self.base_dataset.__getitem__(index, context)
        if batch is None:
            return None

        path = batch.get(self.path_name)

        assert path is not None, f'missing dataset key={self.path_name}'
        assert isinstance(path, str)
        assert os.path.exists(path), f'path={path} does not exist!'

        image_paths = []
        for ext in self.nifti_extensions:
            additional_images = glob(os.path.join(path, '*' + ext))
            image_paths += additional_images
        logger.info(f'Reading NIFTI={image_paths}')
        images = {p: self.nifti_loader(p) for p in image_paths}

        if self.nifty_postprocessing is not None:
            images = self.nifty_postprocessing(images, batch)

        image_n = 0
        for image_name, image in images.items():
            tags = self.volume_serializer(volume=image, header={}, index=image_n)
            for name, value in tags.items():
                assert name not in batch, f'tag collision, tag={name}'
                batch[image_name + name] = value

            image_n += 1

        if self.transform is not None:
            batch = self.transform(batch)
        return batch
