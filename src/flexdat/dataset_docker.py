import fcntl
import logging
import os
import shutil
import subprocess
import sys
import time
from copy import copy
from typing import Any, Callable, Dict, Optional, Union

from .dataset import CoreDataset
from .types import Batch

logger = logging.getLogger(__name__)


try:
    # optional
    import docker

    _has_docker = True
except:
    _has_docker = False


class Locker:
    # protect simultaneous docker executions (e.g., when using GPU)
    def __init__(self, lock_name: Optional[str]) -> None:
        self.lock_name = lock_name

    def __enter__(self) -> None:
        if self.lock_name is not None:
            self.fp = open(self.lock_name)
            fcntl.flock(self.fp.fileno(), fcntl.LOCK_EX)

    def __exit__(self, _type: Any, value: Any, tb: Any) -> None:
        if self.lock_name is not None:
            fcntl.flock(self.fp.fileno(), fcntl.LOCK_UN)
            self.fp.close()


class DatasetDocker(CoreDataset):
    """
    Run a docker container and import the result into the dataset

    parameters:
        container_locations: indicate where to pull the docker image from (<docker_image_name>, <path>)
        setup_index_for_docker_fn: setup required for the docker (e.g., create input folder, output folder, ...).
            Returns the docker kwargs to run the instance
        retrieve_docker_result_fn: load results of the container to be added to the dataset
        gpu_device: the device ID (`0`, `1`, ... `N`) or None for CPU. If this is a callable
            (for example GPU defined by process ID), it will be called before assignment
    """

    def __init__(
        self,
        base_dataset: CoreDataset,
        setup_index_for_docker_fn: Callable[[Batch, str], Dict],
        retrieve_docker_result_fn: Callable[[Batch, str], Dict],
        docker_image_name: str,
        docker_entrypoint: Optional[str] = None,
        gpu_device: Optional[Union[str, Callable[[], str]]] = None,
        container_locations: Optional[Dict] = None,
        tmp_location: Optional[str] = '~/tmp/dataset_docker',
        transform: Optional[Callable[[Batch], Batch]] = None,
        delete_tmp_location: bool = True,
    ):
        super().__init__()

        if not _has_docker:
            logger.error('Docker python API missing! Install it with `pip install docker` for this optional component!')
            raise RuntimeError(
                'Docker python API is not installed! Install it with `pip install docker` for this optional component!'
            )

        self.base_dataset = base_dataset
        self.setup_index_for_docker_fn = setup_index_for_docker_fn
        self.retrieve_docker_result_fn = retrieve_docker_result_fn
        self.docker_image_name = docker_image_name
        if callable(gpu_device):
            # GPU per dataset or process?
            gpu_device = gpu_device()
        self.gpu_device = gpu_device
        self.container_locations = container_locations
        if tmp_location is None:
            tmp_location = '.'
        tmp_location = os.path.expanduser(tmp_location)
        if os.path.expanduser('~') not in tmp_location:
            logging.warning(
                'BEWARE: Docker installed from SNAP mapped directories MUST be inside home directory'
                ' (else directories will be empty!)'
            )
        self.tmp_location = tmp_location
        os.makedirs(tmp_location, exist_ok=True)
        assert tmp_location, 'TODO: temporary file to be handled!'
        self.delete_tmp_location = delete_tmp_location
        self.docker_entrypoint = docker_entrypoint
        self.transform = transform
        self.run_id = 0

        # create the global file lock: this is safe: at this point
        # multiprocessing has not started yet!
        lockname = self._lockname()
        if lockname is not None:
            with open(lockname, 'wb'):
                pass

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _lockname(self) -> Optional[str]:
        return (
            os.path.join(self.tmp_location, self.docker_image_name + f'_gpu_{self.gpu_device}.lck')
            if self.gpu_device is not None
            else None
        )

    def __getitem__(self, index: int, context: Optional[Dict] = None) -> Optional[Batch]:
        logger.info(f'processing index={index}')
        batch = self.base_dataset.__getitem__(index, context)
        if batch is None:
            return None
        batch = copy(batch)

        assert len(self.tmp_location) > 3, 'path too short!!!!!'
        os.makedirs(self.tmp_location, exist_ok=True)
        process_id = str(os.getpid())
        # each process should have its own docker context in case multi-processing is used
        docker_context_path = os.path.join(self.tmp_location, process_id + '_run_' + str(self.run_id))
        self.run_id += 1

        if os.path.exists(docker_context_path):
            # if we cant remove folder, something is odd, raise exception
            shutil.rmtree(docker_context_path)
        os.makedirs(docker_context_path)

        time_start = time.perf_counter()
        client = docker.from_env()
        try:
            # find the target image
            image = client.images.get(self.docker_image_name)
        except docker.errors.ImageNotFound:
            logger.info(
                f'docker image={self.docker_image_name} not found in the docker registry!'
                f' Registry={self.container_locations}'
            )
            image = None

        if image is None:
            # the image is not present in the docker repository. Import it
            assert self.container_locations is not None
            image_path = self.container_locations.get(self.docker_image_name)
            logger.info(f'docker image={self.docker_image_name} found here={image_path}. Importing to docker!')
            assert image_path is not None, f'unknown image={self.docker_image_name}!'
            assert os.path.exists(image_path), f'docker image not found={image_path}'
            try:
                with open(image_path, 'rb') as f:
                    images = client.images.load(f)
                assert len(images) == 1, f'expected single image, got={len(images)}'
                image = images[0]
            except Exception as e:
                # there is nothing we can do here. Terminate the program
                logger.error(f'Docker image import failed! E={e}')
                sys.exit(1)
            logger.info('docker image imported successfully!')

        # run docker
        lockname = self._lockname()
        with Locker(lockname):
            docker_kwargs = self.setup_index_for_docker_fn(batch, docker_context_path)
            assert isinstance(docker_kwargs, dict)  # must be a dictionary!
            if self.gpu_device is not None:
                docker_kwargs['device_requests'] = [
                    docker.types.DeviceRequest(device_ids=[self.gpu_device], capabilities=[['gpu']])
                ]
            if self.docker_entrypoint is not None:
                docker_kwargs['entrypoint'] = self.docker_entrypoint

            try:
                o = client.containers.run(image, **docker_kwargs, stderr=True)
                logger.info(f'container output={o.decode()}')
            except docker.errors.ContainerError as e:
                logger.error(f'Docker container failed to process the input. Error={e}')

            # retrieve results
            batch = self.retrieve_docker_result_fn(batch, docker_context_path)

            if self.delete_tmp_location:
                try:
                    # clean up so files dont accumulate
                    shutil.rmtree(docker_context_path, ignore_errors=True)
                except OSError as e:
                    logger.warning(f'Failed to delete temporary data={docker_context_path}, e={e}')

        time_end = time.perf_counter()
        logger.info(f'processing done! Time={time_end - time_start}')

        if self.transform is not None:
            batch = self.transform(batch)
        return batch
