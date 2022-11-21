import numpy as np
import pytest
from anndata import AnnData

from dance.data import Data

X = np.array([[0, 1], [1, 2], [2, 3]])
Y = np.array([[0], [1], [2]])


def test_data_basic_properties(subtests):
    xad = AnnData(X=X)
    yad = AnnData(X=Y)

    with subtests.test("No training splits"):
        data = Data(x=xad, y=yad)
        assert data.num_cells == 3
        assert data.num_features == 2
        assert data.cells == ["0", "1", "2"]
        assert data.train_idx is data.val_idx is data.test_idx is None

    with subtests.test("Training and testing splits"):
        data = Data(x=xad, y=yad, train_size=2)
        assert data.train_idx == ["0", "1"]
        assert data.val_idx is None
        assert data.test_idx == ["2"]

        data = Data(x=xad, y=yad, train_size=-1, test_size=1)
        assert data.train_idx == ["0", "1"]
        assert data.val_idx is None
        assert data.test_idx == ["2"]

    with subtests.test("Training validation and testing splits"):
        data = Data(x=xad, y=yad, train_size=1, val_size=1)
        assert data.train_idx == ["0"]
        assert data.val_idx == ["1"]
        assert data.test_idx == ["2"]

    with subtests.test("Error sizes"):
        with pytest.raises(TypeError):
            Data(x=xad, y=yad, train_size="1")
        with pytest.raises(ValueError):  # cannot have two -1
            Data(x=xad, y=yad, train_size=-1)
        with pytest.raises(ValueError):  # train size exceeds data size
            Data(x=xad, y=yad, train_size=5)
        with pytest.raises(ValueError):  # sum of sizes exceeds data size
            Data(x=xad, y=yad, train_size=2, test_size=2)
