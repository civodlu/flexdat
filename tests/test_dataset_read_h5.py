import os
import tempfile

import h5py
import numpy as np

from flexdat import DatasetPath, DatasetReadH5


def test_read_h5_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, 'p0.f5')
        with h5py.File(filename, 'w') as f:
            f.create_dataset('value_0', data=42)
            f.create_dataset('value_1', data='str_42')
            f.create_dataset('value_2', data=np.asarray([0, 1, 2, 3]))

        dataset = DatasetPath([filename])
        dataset = DatasetReadH5(dataset)

        batch = dataset[0]
        assert len(batch) == 4
        assert batch['path'] == filename
        assert batch['value_0'] == 42
        assert batch['value_1'] == 'str_42'
        assert (batch['value_2'] == (0, 1, 2, 3)).all()

        dataset = DatasetPath([filename])
        dataset = DatasetReadH5(dataset, keys=('value_0',))
        batch = dataset[0]
        assert len(batch) == 2
        assert batch['path'] == filename
        assert batch['value_0'] == 42
