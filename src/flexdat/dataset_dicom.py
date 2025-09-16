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


def itk_serializer(volume: sitk.Image, header: Optional[Header], index: int, base_name: str = '') -> Dict:
    """
    Serialize the ITK image and header to a Batch.

    Parameters:
        index: the image index among a given batch

    See `extract_itk_image_from_batch` for the reverse operation
    """
    voxels = sitk.GetArrayFromImage(volume)
    if voxels.dtype == np.float64:
        # we generally don't need that much precision
        # convert to float32
        voxels = voxels.astype(np.float32)

    data = {}
    if header is not None:
        for name, value in header.items():
            data[f'{base_name}{name}'] = value

    data[f'{base_name}voxels'] = voxels
    data[f'{base_name}direction'] = np.asarray(volume.GetDirection()).reshape((3, 3))
    data[f'{base_name}spacing'] = np.asarray(volume.GetSpacing(), dtype=np.float32)
    data[f'{base_name}origin'] = np.asarray(volume.GetOrigin(), dtype=np.float32)
    return data


def extract_itk_image_from_batch(batch: Batch, base_name: str, dtype: Optional[np.dtype] = None) -> sitk.Image:
    """
    From a batch, try to recover an ITK image.

    This is the reverse of `itk_serializer`
    """
    voxels = batch.get(base_name + 'voxels')
    assert voxels is not None, f'cannot find voxels in batch name={base_name + "voxels"}! Got keys={batch.keys()}'
    assert len(voxels.shape) == 3, f'must be a DHW shape! Got={voxels.shape}'
    if isinstance(voxels, torch.Tensor):
        voxels = voxels.detach().cpu().numpy()

    if dtype is not None:
        voxels = voxels.astype(dtype)

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


def path_reader_dicom(
    path: str,
    tag_recoder: TagRecorder = default_tag_recorder,
    image_namer: Callable[[Header], str] = lambda h: h['SeriesInstanceUID'],
) -> Tuple[Dict[str, sitk.Image], Dict[str, Any]]:
    """
    Read a folder with possibly multiple DICOM series

    Returns images and selected metadata
    """
    images, headers = read_dicom_folder(path)

    images_final = {}
    headers_final = {}
    for i, h in zip(images, headers):
        uid = image_namer(h)
        tags = tag_recoder(h)
        assert (
            uid not in images_final
        ), f'image UID is duplicated! UID should be unique but another image has the same. Got={uid}'
        images_final[uid] = i
        headers_final[uid] = tags
    return images_final, headers_final
