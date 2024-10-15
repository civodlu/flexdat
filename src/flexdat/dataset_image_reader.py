import logging
import os
from glob import glob
from typing import Any, Callable, Dict, Optional, Sequence, Union

import SimpleITK as sitk

from .dataset import CoreDataset
from .dataset_dicom import VolumeSerializer, itk_serializer
from .dataset_image_folder import ImageLoader
from .dataset_image_processing import (
    ImagePostprocessor,
    image_postprocessing_rename,
    image_postprocessing_rename_fixed,
)
from .itk import read_nifti
from .types import Batch

logger = logging.getLogger(__name__)


def read_path_or_path_sequence_or_folder(
    path: Union[str, Sequence[str]],
    image_loader: ImageLoader = read_nifti,
    dict_name_suffix: str = '_',
    folder_file_extensions: Sequence[str] = ('.nii.gz',),
    # folder_file_renaming: Optional[ImagePostprocessor] = image_postprocessing_rename,
) -> Dict[str, sitk.Image]:
    """
    Read a file or a list of files or a dict of files
    """
    logger.info(f'Reading image={path}')
    if isinstance(path, str):
        if os.path.isfile(path):
            # single image
            return {path: image_loader(path)}
        else:
            # a folder
            image_paths = []
            for ext in folder_file_extensions:
                additional_images = glob(os.path.join(path, '*' + ext))
                image_paths += additional_images
            images = {p: image_loader(p) for p in image_paths}

    if isinstance(path, dict):
        # dict of images
        images = {name + dict_name_suffix: image_loader(p) for name, p in path.items()}
        return images

    if isinstance(path, (list, tuple)):
        image_paths = path  # type: ignore
        images = {p: image_loader(p) for p in image_paths}
        return images

    raise ValueError(f'unsupported path type={path}')


class DatasetImageReader(CoreDataset):
    """
    Dataset pointing to a local path, containing a single image, a list of images

    Example:

    >>> paths = ['/path/1/volume.nii.gz',]
    >>> dataset = DatasetPath(paths)
    >>> dataset = DatasetImageReader(dataset)
    >>> batch = dataset[0]
    """

    def __init__(
        self,
        base_dataset: CoreDataset,
        path_reader: Callable[[Any], Dict[str, sitk.Image]] = read_path_or_path_sequence_or_folder,
        image_postprocessing: Optional[ImagePostprocessor] = image_postprocessing_rename_fixed,
        path_name: str = 'path',
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
        self.image_postprocessing = image_postprocessing
        self.path_name = path_name
        self.base_dataset = base_dataset
        self.volume_serializer = volume_serializer
        self.transform = transform
        self.path_reader = path_reader

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int, context: Optional[Dict] = None) -> Optional[Batch]:
        batch = self.base_dataset.__getitem__(index, context)
        if batch is None:
            return None

        path = batch.get(self.path_name)
        assert path is not None, f'missing dataset key={self.path_name}'

        images = self.path_reader(path)
        if self.image_postprocessing is not None:
            images = self.image_postprocessing(images, batch)

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
