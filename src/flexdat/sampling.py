from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from mypy_extensions import NamedArg

from .dataset_safe import DatasetExceptionDiscard
from .types import Batch, OptionalShapeZYX, ShapeZYX


class CoordinateSampler:
    """
    From a 3D shape and a list of excluded slices, return `N` bounding boxes
    """

    def __call__(self, shape: OptionalShapeZYX, **kwargs: Any) -> Tuple[List[OptionalShapeZYX], List[OptionalShapeZYX]]:
        raise NotImplementedError()


class CoordinateSamplerSlices(CoordinateSampler):
    """
    Sample random 2.5d slices
    """

    def __init__(
        self,
        nb_slices: int,
        nb_samples: int = 1,
        nb_tries: int = 50,
        discard_top_k_slices: int = 0,
        discard_bottom_k_slices: int = 0,
    ) -> None:
        self.nb_slices = nb_slices
        self.nb_tries = nb_tries
        self.nb_samples = nb_samples
        self.discard_top_k_slices = discard_top_k_slices
        self.discard_bottom_k_slices = discard_bottom_k_slices

    def __call__(self, shape: OptionalShapeZYX, **kwargs: Any) -> Tuple[List[OptionalShapeZYX], List[OptionalShapeZYX]]:
        half_slices = self.nb_slices // 2
        nb_slices = shape[0]
        assert nb_slices is not None
        excluded_slice_indices: Sequence[int] = ()

        min_voxels_zyx_included = []
        max_voxels_zyx_included = []
        for i in range(self.nb_samples):
            for _ in range(50):  # have a maximum number of trials
                # randomly sample a slice and check none of the slices
                # belong to the forbidden list of indices
                #
                # NOTE: top slices and bottom slices may be discarded as
                # they often contain much higher noise
                slice_index = np.random.randint(
                    half_slices + self.discard_top_k_slices, nb_slices - half_slices - self.discard_bottom_k_slices
                )
                slice_index_good = True

                # stick with `simple range` notation for indexing with
                # h5 files. the `range(min_index, max_index)` notation
                # is very slow as this is considered `fancy` indexing
                # https://docs.h5py.org/en/latest/high/dataset.html#fancy-indexing
                # https://stackoverflow.com/questions/21766145/h5py-correct-way-to-slice-array-datasets
                slice_range_min = slice_index - half_slices
                slice_range_max = slice_index + half_slices + 1
                for ii in range(slice_range_min, slice_range_max):
                    if ii in excluded_slice_indices:
                        slice_index_good = False
                        break
                if slice_index_good:
                    break
            min_voxels_zyx_included.append((slice_index - half_slices, None, None))
            max_voxels_zyx_included.append((slice_index + half_slices, None, None))

        return min_voxels_zyx_included, max_voxels_zyx_included  # type: ignore


class CoordinateSamplerBlock(CoordinateSampler):
    """
    Sample random 3d blocks
    """

    def __init__(
        self,
        block_shape: ShapeZYX,
        nb_samples: int = 1,
        nb_tries: int = 50,
        discard_top_k_slices: int = 0,
        discard_bottom_k_slices: int = 0,
    ) -> None:
        self.block_shape = block_shape
        self.nb_tries = nb_tries
        self.nb_samples = nb_samples
        self.discard_top_k_slices = discard_top_k_slices
        self.discard_bottom_k_slices = discard_bottom_k_slices

    def __call__(self, shape: OptionalShapeZYX, **kwargs: Any) -> Tuple[List[OptionalShapeZYX], List[OptionalShapeZYX]]:
        assert self.block_shape[0] is not None
        assert self.block_shape[1] is not None
        assert self.block_shape[2] is not None
        assert shape[0] is not None
        assert shape[1] is not None
        assert shape[2] is not None
        nb_slices = shape[0]

        excluded_slice_indices: Sequence[int] = ()

        min_voxels_zyx_included = []
        max_voxels_zyx_included = []
        for i in range(self.nb_samples):
            for _ in range(50):  # have a maximum number of trials
                # randomly sample a slice and check none of the slices
                # belong to the forbidden list of indices
                #
                # NOTE: top slices and bottom slices may be discarded as
                # they often contain much higher noise
                slice_index = np.random.randint(
                    self.discard_top_k_slices, nb_slices - self.block_shape[0] - self.discard_bottom_k_slices
                )
                slice_index_good = True

                # stick with `simple range` notation for indexing with
                # h5 files. the `range(min_index, max_index)` notation
                # is very slow as this is considered `fancy` indexing
                # https://docs.h5py.org/en/latest/high/dataset.html#fancy-indexing
                # https://stackoverflow.com/questions/21766145/h5py-correct-way-to-slice-array-datasets
                slice_range_min = slice_index
                slice_range_max = slice_index + self.block_shape[0]

                y_min = np.random.randint(0, shape[1] - self.block_shape[1])
                x_min = np.random.randint(0, shape[2] - self.block_shape[2])
                for ii in range(slice_range_min, slice_range_max):
                    if ii in excluded_slice_indices:
                        slice_index_good = False
                        break
                if slice_index_good:
                    break
            min_voxels_zyx_included.append((slice_range_min, y_min, x_min))
            max_voxels_zyx_included.append(
                (slice_range_max - 1, y_min + self.block_shape[1] - 1, x_min + self.block_shape[2] - 1)
            )

        return min_voxels_zyx_included, max_voxels_zyx_included  # type: ignore


BlockExtractor = Callable[
    [
        NamedArg(Any, 'blob'),
        NamedArg(str, 'feature_name'),
        NamedArg(OptionalShapeZYX, 'min_voxels_zyx_included'),
        NamedArg(OptionalShapeZYX, 'max_voxels_zyx_included'),
    ],
    np.ndarray,
]


class BlockExtractorH5Data:
    def __call__(
        self,
        blob: h5py.File,
        feature_name: str,
        min_voxels_zyx_included: OptionalShapeZYX,
        max_voxels_zyx_included: OptionalShapeZYX,
    ) -> np.ndarray:
        if (
            min_voxels_zyx_included[1] is None
            and min_voxels_zyx_included[2] is None
            and min_voxels_zyx_included[0] is not None
            and min_voxels_zyx_included[0] >= 0
        ):
            # slice extraction
            assert min_voxels_zyx_included[0] is not None
            assert max_voxels_zyx_included[0] is not None
            assert max_voxels_zyx_included[1] is None
            assert max_voxels_zyx_included[2] is None
            return blob[feature_name][min_voxels_zyx_included[0] : max_voxels_zyx_included[0] + 1]  # type: ignore

        # 3D block extraction
        assert None not in min_voxels_zyx_included
        assert None not in max_voxels_zyx_included
        assert min_voxels_zyx_included[0] >= 0  # type: ignore
        assert min_voxels_zyx_included[1] >= 0  # type: ignore
        assert min_voxels_zyx_included[2] >= 0  # type: ignore

        b = blob[feature_name][
            min_voxels_zyx_included[0] : max_voxels_zyx_included[0] + 1,  # type: ignore
            min_voxels_zyx_included[1] : max_voxels_zyx_included[1] + 1,  # type: ignore
            min_voxels_zyx_included[2] : max_voxels_zyx_included[2] + 1,  # type: ignore
        ]

        expected_shape = (
            1,
            max_voxels_zyx_included[0] - min_voxels_zyx_included[0] + 1,  # type: ignore
            max_voxels_zyx_included[1] - min_voxels_zyx_included[1] + 1,  # type: ignore
            max_voxels_zyx_included[2] - min_voxels_zyx_included[2] + 1,  # type: ignore
        )
        assert b.shape[0] == expected_shape[1]
        assert b.shape[1] == expected_shape[2]
        assert b.shape[2] == expected_shape[3]
        return b.reshape(expected_shape)  # type: ignore


class SamplerH5:
    def __init__(
        self,
        coordinate_sampler: CoordinateSampler = CoordinateSamplerSlices(nb_slices=3),
        block_extractor: BlockExtractor = BlockExtractorH5Data(),
        keys: Optional[Sequence[str]] = None,
        use_context_sync: bool = True,
    ) -> None:
        self.coordinate_sampler = coordinate_sampler
        self.block_extractor = block_extractor
        self.keys = keys
        self.use_context_sync = use_context_sync

    def __call__(self, data: h5py.File, context: Optional[Dict] = None) -> Batch:
        """
        Sample from partially loading the H5.

        Returns batch of data loaded
        """
        if self.keys is None:
            keys_for_sampling: Sequence[str] = []
            for key in data.keys():
                data_value = data[key]
                if len(data_value.shape) >= 3 and '_bounding_boxes' not in key:
                    keys_for_sampling.append(key)  # type: ignore
        else:
            keys_for_sampling = self.keys

        batch = {}

        target_shape_zyx = None
        indices_min_list_zyx = None
        indices_max_list_zyx = None
        if context and self.use_context_sync:
            context_sampling_keys = context.get('sampling_keys')
            if context_sampling_keys and context_sampling_keys == keys_for_sampling:
                # seems like we should not randomly sample but instead
                # use the context info
                target_shape_zyx = context.get('sampling_target_shape_zyx')
                indices_min_list_zyx = context.get('sampling_indices_min_zyx')
                indices_max_list_zyx = context.get('sampling_indices_max_zyx')

        for key in keys_for_sampling:
            data_value = data[key]
            if target_shape_zyx is None:
                target_shape_zyx = data_value.shape
                # TODO: handle slices that should not be sampled
                indices_min_list_zyx, indices_max_list_zyx = self.coordinate_sampler(data_value.shape, data=data)
            else:
                if target_shape_zyx != data_value.shape:
                    # this is a common preprocessing issue. Make this error discardable using `DatasetSafe`
                    raise DatasetExceptionDiscard(
                        f'This sampler can only sample from volumes with the same shape. Got={data_value.shape},'
                        f' target={target_shape_zyx}. File={data.filename}'
                    )

            blocks = []
            assert indices_min_list_zyx is not None
            assert indices_max_list_zyx is not None
            for indices_min, indices_max in zip(indices_min_list_zyx, indices_max_list_zyx):
                block = self.block_extractor(
                    blob=data,
                    feature_name=key,
                    min_voxels_zyx_included=indices_min,
                    max_voxels_zyx_included=indices_max,
                )
                blocks.append(block)

            batch[key] = np.stack(blocks)
            # record the indices: useful if we want to
            # sample a related volume at the same location
            batch['sampling_indices_min_zyx'] = indices_min_list_zyx
            batch['sampling_indices_max_zyx'] = indices_max_list_zyx

        if context is not None:
            # Most likely, we want to sample exactly at the same
            # locations for different volumes. Record where we
            # sampled
            context['sampling_indices_min_zyx'] = indices_min_list_zyx
            context['sampling_indices_max_zyx'] = indices_max_list_zyx
            context['sampling_target_shape_zyx'] = target_shape_zyx
            context['sampling_keys'] = keys_for_sampling
        return batch
