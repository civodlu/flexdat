import logging
import os
from typing import Callable, Dict, Optional

from .dataset import CoreDataset
from .dataset_dicom import VolumeSerializer, itk_serializer
from .dataset_image_folder import ImageLoader
from .dataset_image_processing import (
    ImagePostprocessor,
    image_postprocessing_rename_fixed,
)
from .itk import read_nifti
from .types import Batch

logger = logging.getLogger(__name__)


class DatasetImageReaderDict(CoreDataset):
    """
    Dataset loading images from a dictionary

    Example:

    >>> paths = [{'img1': '/path/1/volume1.nii.gz', 'img2': '/path/1/volume2.nii.gz'},]
    >>> dataset = DatasetPath(paths)
    >>> dataset = DatasetImageReaderDict(dataset)
    >>> batch = dataset[0]
    """

    def __init__(
        self,
        base_dataset: CoreDataset,
        image_loader: ImageLoader = read_nifti,
        image_postprocessing: Optional[ImagePostprocessor] = image_postprocessing_rename_fixed,
        path_name: str = 'path',
        volume_serializer: VolumeSerializer = itk_serializer,
        transform: Optional[Callable[[Batch], Batch]] = None,
        name_postfix: str = '_',
    ):
        """
        Args:
            base_dataset: the base dataset to load DICOM from
            path_name: location name in the base dataset
            volume_serializer: extract information from the image (e.g., voxel, coordinate system)
        """
        super().__init__()
        self.image_loader = image_loader
        self.image_postprocessing = image_postprocessing
        self.path_name = path_name
        self.base_dataset = base_dataset
        self.volume_serializer = volume_serializer
        self.transform = transform
        self.name_postfix = name_postfix

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int, context: Optional[Dict] = None) -> Optional[Batch]:
        batch = self.base_dataset.__getitem__(index, context)
        if batch is None:
            return None

        path = batch.get(self.path_name)

        assert path is not None, f'missing dataset key={self.path_name}'
        assert isinstance(path, dict), 'must be a dictionary!'

        for k, p in path.items():
            assert isinstance(p, str)
            assert os.path.exists(p), f'k={k} path={p} does not exist!'
            assert os.path.isfile(p), f'k={k} path={p} must be a regular file!'
        image_paths = path  # type: ignore

        logger.info(f'Reading image={image_paths}')
        images = {name + self.name_postfix: self.image_loader(path) for name, path in image_paths.items()}

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
