import logging
from typing import Any, Dict, Optional, Sequence

from .dataset import CoreDataset
from .types import Batch

logger = logging.getLogger(__name__)


class DatasetExceptionDiscard(Exception):
    """
    Specific exception to be raised in order to be caught
    by the default `DatasetSafe` exception handler.
    """

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class DatasetSafe(CoreDataset):
    """
    Sometimes, it is not possible to know in advance a case (index)
    is flawed (e.g., realtime data, augmentation, data failure...). In this case
    we want to `swallow the exception` and discard this index.

    If an exception is caught and to be swallowed, the returned batch will be `None`. The dataloader should have
    a special collate_fn to handle the `None` batches.

    Example:

    >>> paths = ['/path/1', '/path/2', '/path/3']
    >>> dataset = DatasetPath(paths)
    >>> dataset = DatasetSafe(dataset)
    >>> assert dataset[3]['path'] == '/path/1'
    """

    def __init__(self, base_dataset: CoreDataset, exceptions_to_catch: Sequence[Any] = (DatasetExceptionDiscard,)):
        super().__init__()
        self.base_dataset = base_dataset
        self.exceptions_to_catch = exceptions_to_catch

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int, context: Optional[Dict] = None) -> Optional[Batch]:
        try:
            batch = self.base_dataset.__getitem__(index, context)
        except Exception as e:
            if type(e) in self.exceptions_to_catch:
                logger.error(f'exception caught index={index}')
                logger.exception(e)
                return None
            else:
                # the exception is not handled, re-raise
                raise e

        # happy path
        return batch
