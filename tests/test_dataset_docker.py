import os
import platform
import tempfile
import time
from functools import partial

import docker
import pytest

from flexdat import DatasetDocker, DatasetPath
from flexdat.dataset_dicom import path_reader_dicom
from flexdat.dataset_docker_utils import read_output_nifti, write_context_nifti
from flexdat.dataset_image_reader import DatasetImageReader

here = os.path.abspath(os.path.dirname(__file__))
docker_dir = os.path.join(here, 'docker')


def make_image(client, tag):
    # programatically build the image and return it
    output = client.api.build(
        path=docker_dir,
        tag=tag,
        dockerfile=os.path.join(docker_dir, 'Dockerfile'),
        buildargs={'UID': '1000', 'GID': '1000'},
        # user='1000:1000'
        # uid='1000',
        # gid='1000',
    )
    for l_out in output:
        print(l_out)

    time.sleep(0.5)  # make sure the image has been registered... ugly :(
    image = client.images.get(tag)
    return image


# docker binded folders MUST be in `~`
docker_tmp_path = os.path.expanduser('~/tmp/dataset_docker')


@pytest.mark.skipif(platform.system() != 'Linux', reason='Linux only!')
def test_make_container_and_run_dummy_test():
    #
    # REQUIRES:
    # - docker installed
    # - user be in dockergroup
    #
    # on a local machine: sudo gpasswd -a $USER docker
    #                     sudo usermod -aG docker $USER
    # if error <permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock>
    #       sudo chmod 666 /var/run/docker.sock
    #
    # issues with docker-ce corrupted (GPU could not be started within container)
    #   reinstall docker: sudo apt-get install --reinstall docker-ce
    #

    # test if we can run a dummy image. This is mostly
    # to help debugging setup issues
    client = docker.from_env()
    tag = 'flexdat_test_2:latest'
    image = make_image(client, tag)

    os.makedirs(docker_tmp_path, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='flexdat_docker_', dir=docker_tmp_path) as tmp_root:
        inputs = os.path.join(tmp_root, 'inputs')
        os.makedirs(inputs)
        outputs = os.path.join(tmp_root, 'outputs')
        os.makedirs(outputs)

        # build the docker image: it simply copy its /inputs in /outputs
        docker_kwargs = {
            'mem_limit': '3g',
            'volumes': {
                inputs: {'bind': '/inputs/', 'mode': 'ro'},
                outputs: {'bind': '/outputs/', 'mode': 'rw'},
            },
        }

        with open(os.path.join(inputs, 'f0.txt'), 'w') as f:
            f.write('FILE_1')
        with open(os.path.join(inputs, 'f1.txt'), 'w') as f:
            f.write('FILE_2')
        gpu_device = None
        # gpu_device = '0'
        if gpu_device is not None:
            docker_kwargs['device_requests'] = [docker.types.DeviceRequest(device_ids=[gpu_device], capabilities=[['gpu']])]
        client.containers.run(image, **docker_kwargs)

        assert os.path.exists(os.path.join(outputs, 'f0.txt'))
        assert os.path.exists(os.path.join(outputs, 'f1.txt'))


@pytest.mark.skipif(platform.system() != 'Linux', reason='Linux only!')
def test_docker_pipeline():
    #
    # REQUIRES:
    # - docker installed
    # - NVIDIA container toolkit
    #      https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
    #      install section (APT) & configuration & restart services
    # - user be in dockergroup:
    #               sudo gpasswd -a $USER docker
    #               sudo usermod -aG docker $USER
    #               newgrp docker
    # - make sure basic docker command works (docker run hello-world)
    #
    #
    # Potential issues:
    #   - issues with docker-ce corrupted (GPU could not be started within container)
    #       reinstall docker: sudo apt-get install --reinstall docker-ce
    #   - `Got permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock`
    #   - NVidia driver: https://stackoverflow.com/questions/75118992/docker-error-response-from-daemon-could-not-select-device-driver-with-capab
    #                curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey |sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    #                && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
    #                && sudo apt-get update
    #                sudo apt-get install -y nvidia-container-toolkit
    #                sudo nvidia-ctk runtime configure --runtime=docker
    #                sudo systemctl restart docker
    path = os.path.join(here, 'resource/dicom_02')
    dataset = DatasetPath([path])
    dataset = DatasetImageReader(dataset, path_reader=partial(path_reader_dicom, image_namer=lambda h: h['Modality']))

    # build a simple docker image
    # the container will copy the content of /inputs to /outputs
    tag = 'flexdat_test4:latest'
    client = docker.from_env()
    make_image(client, tag)

    gpu_device_get = False

    def get_gpu():
        nonlocal gpu_device_get
        gpu_device_get = True
        import torch

        return '0' if torch.cuda.device_count() else None  # CI/CD machine doesn't have a GPU!

    dataset = DatasetDocker(
        dataset,
        setup_index_for_docker_fn=write_context_nifti,
        retrieve_docker_result_fn=read_output_nifti,
        docker_image_name=tag,
        gpu_device=get_gpu,
        # docker_entrypoint='/bin/sh -c "ls -la /inputs"',
    )

    assert len(dataset) == 1

    r = dataset[0]
    assert 'CT_voxels' in r
    assert 'PT_voxels' in r
    assert 'SEG_voxels' in r

    assert (r['output_ct_voxels'] == r['CT_voxels']).all()
    assert 'output_ct_direction' in r
    assert 'output_ct_spacing' in r
    assert 'output_ct_origin' in r


@pytest.mark.skipif(platform.system() != 'Linux', reason='Linux only!')
def test_docker_pipeline_from_image():
    path = os.path.join(here, 'resource/dicom_02')
    dataset = DatasetPath([path])
    dataset = DatasetImageReader(dataset, path_reader=partial(path_reader_dicom, image_namer=lambda h: h['Modality']))

    # build a simple docker image
    # the container will copy the content of /inputs to /outputs
    tag = 'flexdat_test4:latest'
    client = docker.from_env()
    image = make_image(client, tag)

    image_path = os.path.join(docker_tmp_path, 'flexdat_test4.tar.gz')
    with open(image_path, 'wb') as f:
        for chunk in image.save():
            f.write(chunk)

    # the image MUST exist: we just created it!
    client.images.remove(tag)

    dataset = DatasetDocker(
        dataset,
        setup_index_for_docker_fn=write_context_nifti,
        retrieve_docker_result_fn=read_output_nifti,
        docker_image_name=tag,
        container_locations={tag: image_path},
        # docker_entrypoint='/bin/sh -c "ls -la /inputs"',
    )

    assert len(dataset) == 1

    r = dataset[0]
    assert 'CT_voxels' in r
    assert 'PT_voxels' in r
    assert 'SEG_voxels' in r

    assert (r['output_ct_voxels'] == r['CT_voxels']).all()
    assert 'output_ct_direction' in r
    assert 'output_ct_spacing' in r
    assert 'output_ct_origin' in r
