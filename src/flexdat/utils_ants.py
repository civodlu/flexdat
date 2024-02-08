# flake8: noqa
from .utils_import import optional_import

ants = optional_import('ants')

import numpy as np
import SimpleITK as sitk

# def ants_affine_transform_to_4x4(aff: np.ndarray, fixed: np.ndarray) -> np.ndarray:
#    """
#    Convert an ATNs affine_3x3 and a fixed center of rotation into a 4x4
#    affine matrix
#
#    E.g.,
#    >>> import ants
#    >>> moving_ants = ants.image_read(...)
#    >>> template_ants = ants.image_read(...)
#    >>> registered = ants.registration(template_ants, moving_ants, type_of_transform='Affine')
#    >>> ants_tfm = registered['fwdtransforms'][0]
#    >>> affine_4x4 = ants_affine_transform_to_4x4(ants_tfm.parameters, ants_tfm.fixed_parameters)
#    """
#    mat = np.hstack((np.reshape(aff[:9], (3, 3)), aff[9:].reshape((3, 1))))
#    m_translation = mat[:, 3]
#    mat = np.vstack((mat, [0, 0, 0, 1]))
#
#    m_offset = np.zeros((3,))
#
#    for i in range(3):
#        m_offset[i] = m_translation[i] + fixed[i]
#        for j in range(3):
#            m_offset[i] -= mat[i, j] * fixed[j]
#
#    mat[:3, 3] = m_offset
#    return mat


def itk_to_ants_image(
    image: sitk.Image,
) -> "ants.ANTsImage":  # type: ignore
    """
    Transform an ITK image to an ANTs image keeping all the necessary
    spatial information
    """
    voxels = sitk.GetArrayViewFromImage(image).T
    if voxels.dtype != np.float32:
        voxels = voxels.astype(np.float32)

    i_ants = ants.from_numpy(
        voxels,
        origin=image.GetOrigin(),
        spacing=image.GetSpacing(),
        direction=np.asarray(image.GetDirection()).reshape((3, 3)),
    )
    return i_ants


def ants_to_itk_image(
    image: "ants.ANTsImage",  # type: ignore
) -> sitk.Image:
    """
    Convert an ANTs image back to ITK
    """
    image_itk = sitk.GetImageFromArray(image.numpy().T)
    image_itk.SetOrigin(image.origin)
    image_itk.SetSpacing(image.spacing)
    image_itk.SetDirection(image.direction.reshape(9))
    return image_itk
