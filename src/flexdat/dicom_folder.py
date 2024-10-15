import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pydicom
import SimpleITK as sitk

logger = logging.getLogger(__name__)


class MetadataAdaptor:
    def __init__(self, header: pydicom.Dataset) -> None:
        self.header = header
        self.filename = header.filename

    def __getitem__(self, key: str) -> Any:
        return getattr(self.header, key)


def _sort_images_by_series(dcms: List[pydicom.Dataset], dcms_path: List[str]) -> List[List[str]]:
    assert len(dcms) == len(dcms_path)
    try:
        series_type = dcms[0].SeriesType
        if isinstance(series_type, pydicom.multival.MultiValue):
            series_type = '|'.join(series_type)
    except:
        # not an NM series
        series_type = []

    try:
        if len(dcms) == 1 and dcms[0].Modality == 'SEG':
            return [dcms_path]
    except:
        # not a segmentation object
        pass

    # TODO: GATED
    assert 'GATED' not in series_type

    if 'DYNAMIC' in series_type:
        # dynamic series, need to decode the time slices
        nb_frames = dcms[0].NumberOfTimeSlices
        nb_slices_per_frame = dcms[0].NumberOfSlices
        expected_slices = nb_frames * nb_slices_per_frame

        dcms_time_instance = np.asarray([s.ImageIndex - 1 for s in dcms])
        if len(dcms_time_instance) != len(dcms):
            # unexpected number of slices
            logger.warn(
                f'Unexpected number of DICOM slices. Expected={expected_slices}, got={len(dcms)}. Path={dcms_path[0]}'
            )

        # Image index to slice index
        # https://dicom.innolitics.com/ciods/pet-image/pet-image/00541330
        # Image Index = ((Time Slice Index - 1) * (Number of Slices (0054,0081))) + Slice Index
        sorted_instance_number = np.argsort(dcms_time_instance)
        dcms_path_sorted = np.asarray(dcms_path)[sorted_instance_number][::-1]
        time_slices_names = [
            list(dcms_path_sorted[n : n + nb_slices_per_frame]) for n in range(0, len(dcms), nb_slices_per_frame)
        ]
        return time_slices_names

    elif 'STATIC' in series_type or 'WHOLE' in series_type or len(series_type) == 0:
        # DO NOT RELY ON INSTANCE NUMBER: THIS IS NOT ALWAYS CORRECT ORDERING!!
        # Instead, look at the slice coordinates. The one with the largest
        # variations is the one to order the slices with
        image_positions = np.asarray([s.ImagePositionPatient for s in dcms])
        image_positions_std = np.std(image_positions, axis=0)
        axis_ordering = np.argmax(image_positions_std)
        dcms_instance_number = image_positions[:, axis_ordering]

        sorted_instance_number = np.argsort(dcms_instance_number)
        if axis_ordering == 0:
            # for SAGITALLY acquired images!!!!
            # revert the direction for Left/Right
            # Not clear the reason but if we don't do that, the
            # coordinate system is not correct
            sorted_instance_number = sorted_instance_number[::-1]

        # TODO if axis_ordering == 1 what to do? Fine on some tested images

        return [list(np.asarray(dcms_path)[sorted_instance_number])]

    else:
        raise NotImplementedError(f'Type={dcms[0].SeriesType}')


def reorder_slices_single_series(dicom_names: Sequence[str]) -> Sequence[Sequence[str]]:
    """
    Sort a list of dicom files into sorted slices split by SeriesInstanceUID
    """
    if len(dicom_names) == 1:
        return [dicom_names]

    # reorder the slices as this will influence the orientation
    # of the image :(
    dcms_by_series = defaultdict(list)
    path_by_series = defaultdict(list)

    # restrict to minimal number of tags to accelerate the loading
    # full loading will be done in a second stage
    tags = [
        (0x0020, 0x000E),  # SeriesInstanceUID
        (0x0054, 0x1000),  # SeriesType
        (0x0054, 0x0101),  # NumberOfTimeSlices
        (0x0054, 0x0081),  # NumberOfSlices
        (0x0054, 0x1330),  # ImageIndex
        (0x0008, 0x0060),  # Modality
        (0x0020, 0x0032),  # ImagePositionPatient
    ]
    dcms = [pydicom.dcmread(f, specific_tags=tags) for f in dicom_names]  # type: ignore
    for dcm, dcm_path in zip(dcms, dicom_names):
        dcms_by_series[dcm.SeriesInstanceUID].append(dcm)
        path_by_series[dcm.SeriesInstanceUID].append(dcm_path)

    all_volumes: List[List[str]] = []
    for series_uid, dcms in dcms_by_series.items():
        series_paths = path_by_series[series_uid]

        try:
            volumes = _sort_images_by_series(dcms, series_paths)
            if len(volumes) > 0:
                all_volumes += volumes
        except Exception as e:
            print(f'Series could not be sorted, skipping volume! E={e}')
            logger.error(f'Series could not be sorted, skipping volume! E={e}')
            continue

    return all_volumes


def read_dicom_folder(path: str) -> Tuple[Sequence[sitk.Image], Sequence[MetadataAdaptor]]:
    """
    Read folder possibly containing multiple series and studies. Reconstruct them as 3D volume

    If path is a filepath, reconstruct a single DICOM file (e.g., 3D SPECT may be
    stored as a single DICOM file)
    """
    reader = sitk.ImageSeriesReader()
    if os.path.isdir(path):

        def walk(path: str) -> Sequence[str]:
            # GetGDCMSeriesFileNames can only find a SINGLE series with recursive!
            # so walk the folders
            reader = sitk.ImageSeriesReader()
            dcm_names = []
            for dir_name, _, _ in os.walk(path):
                dcm_names += reader.GetGDCMSeriesFileNames(dir_name, loadSequences=False)
            return dcm_names

        dicom_names = walk(path)

        # VERY VERY IMPORTANT: the slice order MATTERS!!!!!
        # https://stackoverflow.com/questions/41037407/itk-simpleitk-dicom-series-loaded-in-wrong-order-slice-spacing-incorrect
        # so we need to reorder them :(
        assert len(dicom_names) > 0, 'folder does not contain DICOM files!'
        series_names = reorder_slices_single_series(dicom_names)
    else:
        # single DICOM (e.g., SPECT)
        series_names = [[path]]

    images = []
    headers = []
    for series_name in series_names:
        try:
            reader.SetFileNames(series_name)
            image = reader.Execute()

            # handle segmentation objects
            image_shape = image.GetSize()
            if len(image_shape) == 4 and image_shape[-1] == 1:
                image = image[:, :, :, 0]

            images.append(image)
            headers.append(MetadataAdaptor(pydicom.dcmread(series_name[0])))  # type: ignore
        except Exception as e:
            print(f'FAILED to read Series={series_name[0]}, skipping! E={e}')
            logger.error(f'FAILED to read Series={series_name[0]}, skipping! E={e}', exc_info=True)

    return images, headers
