import SimpleITK as sitk
from flexdat.itk import apply_homogeneous_affine_transform, affine_tfm_to_homogenous, affine_transform_4x4_to_itk
import numpy as np
import torch


def test_itk_homogenous():
    tx2 = sitk.AffineTransform(3)
    tx2.SetCenter((56.62767028808594, 8.902861595153809, -83.6222152709961))
    rot = np.array([[ 0.93512523, -0.02702491,  0.05725547],
        [-0.04591934,  0.99985832, -0.06661809],
        [ 0.06233513, -0.02044608,  2.01888144]])
    tx2.SetMatrix(rot.reshape((-1,)))
    tx2.SetTranslation((10.0, 11.0, 12.0))

    p = (1, 2, 3)

    p_tx2 = tx2.TransformPoint(p)
    tx_homogenous = affine_tfm_to_homogenous(tx2)
    p_h4 = apply_homogeneous_affine_transform(tx_homogenous, torch.from_numpy(np.asarray(p))).numpy()
    assert np.abs(p_tx2 - p_h4).max() < 1e-5

    # conversion homogenous <-> ITK
    tx_homogenous_2 = affine_tfm_to_homogenous(affine_transform_4x4_to_itk(torch.from_numpy(tx_homogenous)))
    assert np.abs(tx_homogenous_2 - tx_homogenous).max() < 1e-7
