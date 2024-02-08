import os
import subprocess
from glob import glob
from typing import Any, Dict, Sequence

from .dataset_dicom import extract_itk_image_from_batch, itk_serializer
from .itk import read_nifti, write_nifti
from .types import Batch


def write_context_nifti(
    batch: Batch,
    path: str,
    input_names: Sequence[str] = ('CT_',),
    input_folder: str = 'inputs',
    output_folder: str = 'outputs',
    with_user: bool = False,
    additional_kwargs: Dict[str, Any] = {'mem_limit': '10g', 'ipc_mode': 'host'},
) -> Dict:
    """
    Parameters:
        with_user: if True, the current user will be used to run the container.
            to be used ONLY if the container was created with a user
    """
    for input_name in input_names:
        itk = extract_itk_image_from_batch(batch, base_name=input_name)
        input_root = os.path.join(path, input_folder)
        output_root = os.path.join(path, output_folder)
        os.makedirs(input_root, exist_ok=False)  # `exist_ok` should NEVER happen!
        os.makedirs(output_root, exist_ok=False)
        v_path = os.path.join(input_root, input_name.lower().strip('_')) + '.nii.gz'
        write_nifti(itk, v_path)

    docker_kwargs: Dict = {
        'volumes': {
            input_root: {'bind': f'/{input_folder}', 'mode': 'ro'},
            output_root: {'bind': f'/{output_folder}', 'mode': 'rw'},
        },
        **additional_kwargs,
    }

    if with_user:
        # we need to setup the user within a container to be the same user
        # that runs the container so that files written inside the container
        # can be deleted!
        user_id = subprocess.check_output(['id', '-u']).decode('utf8').replace('\n', '')
        group_id = subprocess.check_output(['id', '-g']).decode('utf8').replace('\n', '')
        user = f'{user_id}:{group_id}'
        docker_kwargs['user'] = user

    return docker_kwargs


def read_output_nifti(
    batch: Batch,
    path: str,
    name_prefix: str = 'output_',
    docker_output_folder: str = 'outputs',
) -> Batch:
    """
    Parameters:
        batch: the current batch
        path: the base path of the docker temporary output
        name_prefix: the prefix to be applied on each volume found in the output
            folder (`docker_output_folder`) of the container

    Returns:
        the updated batch
    """
    nifti_files = glob(os.path.join(path, docker_output_folder, '*.nii*'))
    for f in nifti_files:
        _, filename = os.path.split(f)
        nifty_index = filename.rfind('.nii')
        assert nifty_index >= 0, 'expecting .nii or .nii.gz extension!'
        base_name = filename[:nifty_index] + '_'
        volume = read_nifti(f)
        seg_attributes = itk_serializer(volume, {}, 0, base_name=name_prefix + base_name)
        batch = {**batch, **seg_attributes}

    return batch
