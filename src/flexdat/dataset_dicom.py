import logging
import os
from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
import torch
from mypy_extensions import DefaultNamedArg, NamedArg

from .dataset import CoreDataset
from .dataset_dicom_postprocessor import Header
from .dicom_folder import read_dicom_folder
from .itk import make_sitk_image
from .types import Batch

logger = logging.getLogger(__name__)


DicomLoader = Callable[[str], Tuple[Sequence[sitk.Image], Sequence[Header]]]
TagRecorder = Callable[[Any], Dict[str, Any]]
DicomSorter = Callable[[Sequence[sitk.Image], Sequence[Header]], Dict[str, Tuple[sitk.Image, Header]]]
DicomsPostprocessor = Callable[[Sequence[sitk.Image], Sequence[Header]], Tuple[Sequence[sitk.Image], Sequence[Header]]]


VolumeSerializer = Callable[
    [
        NamedArg(sitk.Image, 'volume'),
        NamedArg(Header, 'header'),
        NamedArg(int, 'index'),
        DefaultNamedArg(str, 'base_name'),
    ],
    Dict,
]

VolumeExtractor = Callable[[Dict, str], sitk.Image]


def default_tag_recorder(header: Any) -> Dict:
    tags = ('SeriesInstanceUID', 'StudyInstanceUID', 'Modality', 'Manufacturer')
    # `str` to remove the pydicom native type: cannot be serialized in H5
    d = {t: str(header[t]) for t in tags}
    return d


def itk_serializer(volume: sitk.Image, header: Header, index: int, base_name: str = '') -> Dict:
    """
    Serialize the ITK image to a Batch.

    Parameters:
        index: the image index among a given batch

    See `extract_itk_image_from_batch` for the reverse operation
    """
    voxels = sitk.GetArrayFromImage(volume)
    if voxels.dtype == np.float64:
        # we generally don't need that much precision
        # convert to float32
        voxels = voxels.astype(np.float32)

    return {
        f'{base_name}voxels': voxels,
        f'{base_name}direction': np.asarray(volume.GetDirection()).reshape((3, 3)),
        f'{base_name}spacing': np.asarray(volume.GetSpacing(), dtype=np.float32),
        f'{base_name}origin': np.asarray(volume.GetOrigin(), dtype=np.float32),
    }


def extract_itk_image_from_batch(batch: Batch, base_name: str) -> sitk.Image:
    """
    From a batch, try to recover an ITK image.

    This is the reverse of `itk_serializer`
    """
    voxels = batch.get(base_name + 'voxels')
    assert voxels is not None, f'cannot find voxels in batch name={base_name + "_voxels"}! Got keys={batch.keys()}'
    assert len(voxels.shape) == 3, f'must be a DHW shape! Got={voxels.shape}'
    if isinstance(voxels, torch.Tensor):
        voxels = voxels.detach().cpu().numpy()

    direction = batch.get(base_name + 'direction')
    if isinstance(direction, torch.Tensor):
        direction = direction.detach().cpu().numpy()
    if direction is None:
        direction = np.eye(3, dtype=float)

    spacing = batch.get(base_name + 'spacing')
    if spacing is None:
        spacing = np.full(3, fill_value=1.0, dtype=float)
    if isinstance(spacing, torch.Tensor):
        spacing = spacing.detach().cpu().numpy()

    origin = batch.get(base_name + 'origin')
    if isinstance(origin, torch.Tensor):
        origin = origin.detach().cpu().numpy()
    if origin is None:
        origin = np.full(3, fill_value=0.0, dtype=float)
    assert direction.shape == (3, 3)
    assert spacing.shape == (3,)
    assert origin.shape == (3,)

    return make_sitk_image(
        voxels,
        origin_xyz=origin.astype(float),
        spacing_xyz=spacing.astype(float),
        direction_xyz=direction.ravel().astype(float),
    )


def sort_dicom_by_modality(volumes: Sequence[sitk.Image], headers: Sequence[Header]) -> Dict[str, Tuple[sitk.Image, Header]]:
    modalities: Dict[str, Any] = {}
    assert len(volumes) == len(headers)
    for v, h in zip(volumes, headers):
        modality = h['Modality']
        assert modality not in modalities, f'Already got modality={modalities}, got={modality}'
        modalities[modality + '_'] = (v, h)

    return modalities


def sort_single_dicom(volumes: Sequence[sitk.Image], headers: Sequence[Header]) -> Dict[str, Tuple[sitk.Image, Header]]:
    assert len(volumes) == 1, f'expected a SINGLE DICOM, got={len(volumes)}'
    return {'': (volumes[0], headers[0])}


class DatasetMultipleDicoms(CoreDataset):
    """
    Dataset pointing to a local path, containing multiple scans of
    the same patient (e.g., PET/CT modality or multiple MR sequences)

    Example:

    >>> paths = ['/path/1', '/path/2']
    >>> dataset = DatasetPath(paths)
    >>> dataset = DatasetMultipleDicoms(dataset)
    >>> batch = dataset[0]

    Example: processing all volumes to the same isotropic 2.0mm spacing

    >>> paths = ['/path/1', '/path/2']
    >>> dataset = DatasetPath(paths)
    >>> dataset = DatasetMultipleDicoms(
            dataset,
            dicoms_postprocessing=partial(post_processor_resample_fixed_spacing,
                geometry_reference_modality='PT', target_spacing_xyz=(2, 2, 2)))
    >>> batch = dataset[0]
    """

    def __init__(
        self,
        base_dataset: CoreDataset,
        dicom_loader: DicomLoader = read_dicom_folder,
        dicoms_postprocessing: Optional[DicomsPostprocessor] = None,
        dicom_sorter: DicomSorter = sort_dicom_by_modality,
        path_name: str = 'path',
        name_prefix: str = '',
        volume_serializer: VolumeSerializer = itk_serializer,
        transform: Optional[Callable[[Batch], Batch]] = None,
        tag_recoder: TagRecorder = default_tag_recorder,
    ):
        """
        Args:
            base_dataset: the base dataset to load DICOM from
            dicom_loader: how to load the DICOM file/directory
            path_name: location name in the base dataset
            volume_serializer: extract information from the image (e.g., voxel, coordinate system)
            tag_recoder: record DICOM tags
            name_prefix: a prefix to be appended
        """

        super().__init__()
        self.dicom_loader = dicom_loader
        self.dicom_sorter = dicom_sorter
        self.dicoms_postprocessing = dicoms_postprocessing
        self.path_name = path_name
        self.base_dataset = base_dataset
        self.tag_recoder = tag_recoder
        self.volume_serializer = volume_serializer
        self.transform = transform
        self.name_prefix = name_prefix

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

        logger.info(f'Reading DICOM={path}')
        images, headers = self.dicom_loader(path)
        if self.dicoms_postprocessing is not None:
            images, headers = self.dicoms_postprocessing(images, headers)

        sorted_images = self.dicom_sorter(images, headers)
        image_n = 0
        for image_name, (image, header) in sorted_images.items():
            tags = self.volume_serializer(volume=image, header=header, index=image_n)
            for name, value in tags.items():
                assert name not in batch, f'tag collision, tag={name}'
                batch[self.name_prefix + image_name + name] = value

            tags = self.tag_recoder(header)
            for name, value in tags.items():
                assert name not in batch, f'tag collision, tag={name}'
                batch[self.name_prefix + image_name + name] = value

            image_n += 1

        if self.transform is not None:
            batch = self.transform(batch)
        return batch


# expecting a single DICOM for each path
DatasetSingleDicom = partial(DatasetMultipleDicoms, dicom_sorter=sort_single_dicom)
