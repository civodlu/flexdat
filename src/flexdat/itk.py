from numbers import Number
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import SimpleITK as sitk
import torch

ItkInterpolatorType = Literal['nearest', 'linear', 'spline', 'gaussian', 'lanczos', 'label_gaussian']
SpacingType = Union[float, Tuple[Optional[float], Optional[float], Optional[float]]]
SizeOptionalType = Tuple[Optional[int], Optional[int], Optional[int]]


def get_itk_interpolator(mode: ItkInterpolatorType = 'linear') -> Any:
    """
    Helper function to map mode and interpolators (e.g., in an external API)
    """
    if mode == 'nearest':
        return sitk.sitkNearestNeighbor
    elif mode == 'linear':
        return sitk.sitkLinear
    elif mode == 'spline':
        return sitk.sitkBSpline
    elif mode == 'gaussian':
        return sitk.sitkGaussian
    elif mode == 'lanczos':
        return sitk.sitkLanczosWindowedSinc
    elif mode == 'label_gaussian':
        return sitk.sitkLabelGaussian
    else:
        raise NotImplementedError(f'interpolator={mode} not handled!')


def is_pixelid_discrete(image: sitk.Image) -> bool:
    """
    Return true if an image has discrete voxel values
    """
    return image.GetPixelID() in [
        sitk.sitkUInt8,
        sitk.sitkInt8,
        sitk.sitkUInt16,
        sitk.sitkInt16,
        sitk.sitkUInt32,
        sitk.sitkInt32,
        sitk.sitkUInt64,
        sitk.sitkInt64,
    ]


def get_sitk_image_attributes(sitk_image: sitk.Image) -> Dict[str, Any]:
    """Get physical space attributes (meta-data) of the image."""
    attributes = {}
    attributes['pixelid'] = sitk_image.GetPixelIDValue()
    attributes['origin'] = sitk_image.GetOrigin()
    attributes['direction'] = sitk_image.GetDirection()
    attributes['spacing'] = np.array(sitk_image.GetSpacing())
    attributes['shape'] = np.array(sitk_image.GetSize(), dtype=int)
    return attributes


def make_sitk_image(
    image: np.ndarray,
    origin_xyz: np.ndarray,
    spacing_xyz: np.ndarray,
    direction_xyz: Tuple[float, ...] = (1, 0, 0, 0, 1, 0, 0, 0, 1),
) -> sitk.Image:
    """
    Create an Simple ITK image from a numpy array

    Returns:
        a sitk image
    """
    assert len(image.shape) == 3
    assert len(origin_xyz) == 3
    assert len(spacing_xyz) == 3
    assert len(direction_xyz) == 9

    image_sitk = sitk.GetImageFromArray(image)
    image_sitk.SetOrigin(origin_xyz)
    image_sitk.SetSpacing(spacing_xyz)
    image_sitk.SetDirection(direction_xyz)

    return image_sitk


def make_sitk_zero_like(image: sitk.Image, pixel_id: Any = None) -> sitk.Image:
    """
    Create a Simple ITK image filled with zero with the same geometry / type
    make_sitk_image

    Returns:
        a sitk image
    """
    new_image = sitk.Image(image.GetSize(), image.GetPixelID() if pixel_id is None else pixel_id)
    new_image.SetOrigin(image.GetOrigin())
    new_image.SetSpacing(image.GetSpacing())
    new_image.SetDirection(image.GetDirection())
    return new_image


def make_sitk_like(image: np.ndarray, target: sitk.Image, dtype: Any = None) -> sitk.Image:
    """
    Create a sitk image geometry based on a given sitk image

    Returns:
        a sitk image
    """
    assert target.GetSize() == image.shape[::-1]
    if dtype is None:
        # default to same type as image
        if image.dtype == bool:
            # however, sitk does not handle `bool` type
            dtype = np.uint8

    if dtype is not None:
        image = image.astype(dtype)

    new_image = sitk.GetImageFromArray(image)
    new_image.SetOrigin(target.GetOrigin())
    new_image.SetSpacing(target.GetSpacing())
    new_image.SetDirection(target.GetDirection())
    return new_image


def resample_spacing(
    image: sitk.Image,
    target_spacing_xyz: SpacingType,
    background_value: float = 0,
    interpolator: ItkInterpolatorType = 'linear',
    segmentation_dtype: Optional[Any] = (np.uint8,),
) -> sitk.Image:
    """
    Resample a given image with a target spacing, keeping the overall geometry similar
    """

    # make sure the segmentations are NOT interpolated!
    volume_dtype = sitk.GetArrayViewFromImage(image).dtype
    if segmentation_dtype is not None and volume_dtype in segmentation_dtype:
        # spacial case: if we have segmentations, do NOT interpolate!
        interpolator_itk = sitk.sitkNearestNeighbor
    else:
        interpolator_itk = get_itk_interpolator(interpolator)

    spacing_xyz = image.GetSpacing()
    shape_xyz = image.GetSize()
    if isinstance(target_spacing_xyz, Number):
        target_spacing_xyz = np.asarray([float(target_spacing_xyz)] * len(shape_xyz))
    else:
        # copy spacing from source for `None`
        target_spacing_xyz = [
            v if v is not None else spacing_xyz[i] for i, v in enumerate(target_spacing_xyz)  # type: ignore
        ]
        target_spacing_xyz = np.asarray(target_spacing_xyz, dtype=float)  # type: ignore
        assert len(target_spacing_xyz) == len(shape_xyz)  # type: ignore

    shape_isotropic_xyz = (np.asarray(spacing_xyz) * np.asarray(shape_xyz) / target_spacing_xyz).round().astype(int)
    shape_isotropic_xyz = [int(s) for s in shape_isotropic_xyz]
    image_isotropic = sitk.Resample(
        image,
        size=shape_isotropic_xyz,  # type: ignore
        interpolator=interpolator_itk,
        outputOrigin=image.GetOrigin(),
        outputSpacing=target_spacing_xyz,
        outputDirection=image.GetDirection(),
        defaultPixelValue=background_value,
    )

    return image_isotropic


def get_itk_size_mm_xyz(image: sitk.Image) -> np.ndarray:
    """
    Return the Field of view size in mm
    """
    return np.asarray(image.GetSize()) * np.asarray(image.GetSpacing(), dtype=np.float32)


def resample_voxels(
    image: sitk.Image,
    target_voxel_xyz: Tuple[Optional[int], Optional[int], Optional[int]],
    background_value: float = 0,
    interpolator: ItkInterpolatorType = 'linear',
    segmentation_dtype: Any = (np.uint8,),
) -> sitk.Image:
    """
    Resample an image such that the field of view is the same but with a different number of voxels
    """
    # make sure the segmentations are NOT interpolated!
    volume_dtype = sitk.GetArrayViewFromImage(image).dtype
    if volume_dtype in segmentation_dtype:
        # spacial case: if we have segmentations, do NOT interpolate!
        interpolator_itk = sitk.sitkNearestNeighbor
    else:
        interpolator_itk = get_itk_interpolator(interpolator)

    shape_xyz = image.GetSize()
    assert len(target_voxel_xyz) == 3
    assert len(shape_xyz) == 3

    target_shape_xyz = np.asarray([v if v is not None else shape_xyz[v_n] for v_n, v in enumerate(target_voxel_xyz)])
    target_spacing_xyz = get_itk_size_mm_xyz(image) / target_shape_xyz

    image_r = sitk.Resample(
        image,
        size=[int(v) for v in target_shape_xyz],  # type: ignore
        interpolator=interpolator_itk,
        outputOrigin=image.GetOrigin(),
        outputSpacing=[float(v) for v in target_spacing_xyz],
        outputDirection=image.GetDirection(),
        defaultPixelValue=background_value,
    )

    return image_r


def resample_like(
    image_source: sitk.Image,
    image_target: sitk.Image,
    interpolator: Optional[ItkInterpolatorType] = None,
    background_value: float = 0,
) -> sitk.Image:
    """
    Resample a source image into the target geometry.
    """
    if interpolator is None:
        if is_pixelid_discrete(image_source):
            # most likely segmentation, no interpolation!
            interpolator_itk = get_itk_interpolator('nearest')
        else:
            interpolator_itk = get_itk_interpolator('spline')
    else:
        interpolator_itk = get_itk_interpolator(interpolator)

    image_source_resampled = sitk.Resample(
        image_source,
        size=image_target.GetSize(),  # type: ignore
        interpolator=interpolator_itk,
        outputOrigin=image_target.GetOrigin(),
        outputSpacing=image_target.GetSpacing(),
        outputDirection=image_target.GetDirection(),
        defaultPixelValue=background_value,
    )

    return image_source_resampled


def standard_orientation(v: sitk.Image) -> sitk.Image:
    """
    MR images may be oriented in any direction (axis swapped or inverted).

    Some algorithms depend on relative positions (e.g., "top" or "right"),
    which is complicated by this arbitrary orientation (voxel grid would not match
    general orientation of the real world grid).

    One possibility is simply to resample the data in a geometry with
    standard axes, now the patient and voxel grids are "correlated"
    (minus small rotations & spacing)

    Calculate bouding box positions and fit them to a standard orientated geometry.
    """
    filter = sitk.DICOMOrientImageFilter()
    filter.SetDesiredCoordinateOrientation('LPS')
    v_standard: sitk.Image = filter.Execute(v)
    return v_standard


def affine_transform_4x4_to_itk(m: torch.Tensor) -> sitk.AffineTransform:
    """
    Express a generic affine transform as an ITK affine transform
    """
    assert m.shape == (4, 4)

    m2 = m.clone()
    m2[:3, :3] = m[:3, :3].T  # weird transpose required. GetParameters() is also transposed!
    column_major = m2[:3].numpy().flatten('F')  # parameters are in column major ordering

    tfm = sitk.AffineTransform(3)
    tfm.SetParameters(tuple(column_major.astype(np.float64)))
    return tfm


def affine_tfm_to_homogenous(tfm: sitk.AffineTransform) -> np.ndarray:
    """
    Extract a 4x4 matrix from an affine transformation

    This is the reverse of affine_transform_4x4_to_itk
    """
    # see discussion here: https://discourse.itk.org/t/express-affinetransform-as-single-4x4-matrix/3193/5
    A = np.array(tfm.GetMatrix()).reshape(3, 3)
    c = np.array(tfm.GetCenter())
    t = np.array(tfm.GetTranslation())
    overall = np.eye(4)
    overall[0:3, 0:3] = A
    overall[0:3, 3] = -np.dot(A, c) + t + c
    return overall


def get_itk_center_mm_xyz(v: sitk.Image) -> torch.Tensor:
    """
    Calculate the center of the volume in mm
    """
    half_index = (np.asarray(v.GetSize()) - 1) / 2.0
    return torch.Tensor(v.TransformContinuousIndexToPhysicalPoint(half_index))


def get_itk_rotation_4x4(v: sitk.Image) -> torch.Tensor:
    """
    Extract the rotational component only of the patient geometry as a 4x4 matrix
    """
    tfm = torch.eye(4, dtype=torch.float32)
    tfm[:3, :3] = torch.asarray(v.GetDirection()).reshape((3, 3))
    return tfm


def get_voxel_to_mm_transform_from_attributes(
    origin_xyz: Tuple[float, float, float], spacing_xyz: Tuple[float, float, float], direction: Tuple[int, ...]
) -> torch.Tensor:
    """
    Return the 4x4 affine matrix that transform a voxel coordinate to world space in mm
    from the ITK attributes.
    """
    pst = np.eye(4, dtype=np.float32)
    pst[:3, 3] = origin_xyz
    r = np.asarray(direction).reshape((3, 3))
    pst[0:3, 0:3] = r

    for n in range(3):
        pst[:3, n] *= spacing_xyz[n]
    return torch.from_numpy(pst)


def get_voxel_to_mm_transform(image: sitk.Image) -> torch.Tensor:
    """
    Return the 4x4 affine matric that transform a voxel coordinate to world space in mm
    """
    return get_voxel_to_mm_transform_from_attributes(
        origin_xyz=image.GetOrigin(), spacing_xyz=image.GetSpacing(), direction=image.GetDirection()
    )


def apply_homogeneous_affine_transform(
    transform: torch.Tensor, position: Union[torch.Tensor, Tuple[float, ...]]
) -> torch.Tensor:
    """
    Apply an homogeneous affine transform (4x4 for 3D or 3x3 for 2D) to a position
    Args:
        transform: an homogeneous affine transformation
        position: XY(Z) position
    Returns:
        a transformed position XY(Z)
    """
    position = torch.tensor(position)
    assert len(transform.shape) == 2
    assert len(position.shape) == 1
    dim = position.shape[0]
    assert transform.shape[0] == transform.shape[1]
    assert transform.shape[0] == dim + 1
    # decompose the transform as a (3x3 transform, translation) components
    position = position.unsqueeze(1).type(torch.float32)
    return np.matmul(transform[:dim, :dim], position).squeeze(1) + transform[:dim, dim]
    # return transform[:dim, :dim].mm(position).squeeze(1) + transform[:dim, dim]


def crop_image(
    v: sitk.Image,
    predicate_fn: Callable[[sitk.Image], np.ndarray] = lambda v: sitk.GetArrayViewFromImage(v) > 0,
    return_inclusive_bounding_box: bool = False,
) -> Union[sitk.Image, Tuple[sitk.Image, np.ndarray, np.ndarray]]:
    """
    Crop the volume so that predicate_fn(v) has minimal size

    Parameters:
        return_inclusive_bounding_box: if True
    """
    v_pred = predicate_fn(v)
    assert v_pred.shape[::-1] == v.GetSize(), f'shape={v_pred.shape[::-1]}, got={v.GetSize()}'

    indices = np.where(v_pred > 0)
    min_indices_zyx = np.min(indices, axis=1)
    max_indices_zyx = np.max(indices, axis=1)

    new_origin = v.TransformIndexToPhysicalPoint([int(i) for i in min_indices_zyx[::-1]])

    v_np = sitk.GetArrayViewFromImage(v)
    v_sub = v_np[
        min_indices_zyx[0] : max_indices_zyx[0] + 1,
        min_indices_zyx[1] : max_indices_zyx[1] + 1,
        min_indices_zyx[2] : max_indices_zyx[2] + 1,
    ]

    cropped_v = make_sitk_image(v_sub, origin_xyz=new_origin, spacing_xyz=v.GetSpacing(), direction_xyz=v.GetDirection())

    if not return_inclusive_bounding_box:
        return cropped_v

    return cropped_v, min_indices_zyx, max_indices_zyx


def read_nifti(path: str) -> sitk.Image:
    """Read a NIfTI image. Return a SimpleITK Image."""
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    nifti: sitk.Image = reader.Execute()

    # nifti = sitk.ReadImage(str(path))
    return nifti


def write_nifti(sitk_img: sitk.Image, path: str) -> None:
    """Save a SimpleITK Image to disk in NIfTI format."""
    writer = sitk.ImageFileWriter()
    writer.SetImageIO("NiftiImageIO")
    writer.SetFileName(str(path))
    writer.Execute(sitk_img)
