import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from dance.data import Data

X = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.float32)
Y = np.array([[0], [1], [2]], dtype=np.float32)


def test_data_basic_properties(subtests):
    adata = AnnData(X=X)

    with subtests.test("No training splits"):
        data = Data(adata.copy())
        assert data.num_cells == 3
        assert data.num_features == 2
        assert data.cells == ["0", "1", "2"]
        assert data.train_idx is data.val_idx is data.test_idx is None

    with subtests.test("All training"):
        data = Data(adata.copy(), train_size="all")
        assert data.train_idx == [0, 1, 2]
        assert data.val_idx is None
        assert data.test_idx is None

    with subtests.test("Training and testing splits"):
        data = Data(adata.copy(), train_size=2)
        assert data.train_idx == [0, 1]
        assert data.val_idx is None
        assert data.test_idx == [2]

        data = Data(adata.copy(), train_size=-1, test_size=1)
        assert data.train_idx == [0, 1]
        assert data.val_idx is None
        assert data.test_idx == [2]

    with subtests.test("Training validation and testing splits"):
        data = Data(adata.copy(), train_size=1, val_size=1)
        assert data.train_idx == [0]
        assert data.val_idx == [1]
        assert data.test_idx == [2]

    with subtests.test("Error sizes"):
        with pytest.raises(TypeError):
            Data(adata.copy(), train_size="1")
        with pytest.raises(ValueError):  # cannot have two -1
            Data(adata.copy(), train_size=-1)
        with pytest.raises(ValueError):  # train size exceeds data size
            Data(adata.copy(), train_size=5)
        with pytest.raises(ValueError):  # sum of sizes exceeds data size
            Data(adata.copy(), train_size=2, test_size=2)

    with subtests.test("Index range dict"):
        split_index_range_dict = {"train": (0, 1), "ref": (0, 2), "inf": (2, 3)}
        data = Data(adata.copy(), split_index_range_dict=split_index_range_dict)

        assert data.train_idx == [0]
        assert data.get_split_idx("ref") == [0, 1]
        assert data.get_split_idx("inf") == [2]

    with subtests.test("Index range dict errors"):
        with pytest.raises(TypeError):  # value must be a two tuple, not three tuple
            Data(adata.copy(), split_index_range_dict={"train": (0, 1, 2)})

        with pytest.raises(TypeError):  # value must be a two tuple, not a list
            Data(adata.copy(), split_index_range_dict={"train": [0, 1]})

        with pytest.raises(TypeError):  # value must be a two tuple of int, not str
            Data(adata.copy(), split_index_range_dict={"train": ("0", "1")})

    with subtests.test("Full split"):
        data = Data(adata.copy(), full_split_name="inference")

        assert data.train_idx is None
        assert data.get_split_idx("inference") == [0, 1, 2]


def test_get_data(subtests):
    adata = AnnData(X=X, obs=pd.DataFrame(X, columns=["a", "b"]), var=pd.DataFrame(X.T, columns=["x", "y", "z"]))
    adata.obsm["feature1"] = X + 10
    adata.obsm["feature2"] = X + 20
    adata.layers["layer_feature"] = X + 30
    adata.obsm["obsm_feature"] = X
    adata.obsp["obsp_feature"] = X @ X.T
    adata.varm["varm_feature"] = X.T
    adata.varp["varp_feature"] = X.T @ X
    adata.obsm["label"] = Y

    with subtests.test("Single feature"):
        data = Data(adata.copy(), train_size=2)
        data.set_config(label_channel="label")

        x_train, y_train = data.get_train_data()
        assert x_train.tolist() == [[0, 1], [1, 2]]
        assert y_train.tolist() == [[0], [1]]

        x_test, y_test = data.get_test_data()
        assert x_test.tolist() == [[2, 3]]
        assert y_test.tolist() == [[2]]

        # Validation set not set
        pytest.raises(KeyError, data.get_val_data)

    with subtests.test("Multi feature"):
        data = Data(adata.copy(), train_size=2)
        data.set_config(feature_channel=[None, "feature1", "feature2"], label_channel="label")

        (x1_train, x2_train, x3_train), y_train = data.get_train_data()
        assert x2_train.tolist() == [[10, 11], [11, 12]]
        assert x3_train.tolist() == [[20, 21], [21, 22]]
        assert y_train.tolist() == [[0], [1]]

    with subtests.test("Multi type feature"):
        data = Data(adata.copy(), train_size=2)
        data.set_config(
            feature_channel=["obsm_feature", "obsp_feature", "varm_feature", "varp_feature", "layer_feature"],
            feature_channel_type=["obsm", "obsp", "varm", "varp", "layers"], label_channel="label")

        (x_obsm, x_obsp, x_varm, x_varp, x_layer), y = data.get_train_data()
        assert x_obsm.tolist() == [[0, 1], [1, 2]]
        assert x_obsp.tolist() == [[1, 2], [2, 5]]
        assert x_varm.tolist() == [[0, 1, 2], [1, 2, 3]]
        assert x_varp.tolist() == [[5, 8], [8, 14]]
        assert x_layer.tolist() == [[30, 31], [31, 32]]
        assert y.tolist() == [[0], [1]]

    with subtests.test("Single column feature"):
        data = Data(adata.copy(), train_size=2)
        data.set_config(feature_channel=["a", "z"], feature_channel_type=["obs", "var"], label_channel="label")

        (x1, x2), _ = data.get_train_data()
        assert x1.tolist() == [0, 1]
        assert x2.tolist() == [2, 3]


def test_append(subtests):
    data1 = Data(AnnData(X=X), train_size=1)
    data2 = Data(AnnData(X=X), train_size=2)
    data2_split_idx = {"train": [0, 1], "test": [2]}

    with subtests.test(mode="merge"):
        data = data1.copy()
        data.append(data2, mode="merge")
        assert data._split_idx_dict == {"train": [0, 3, 4], "test": [1, 2, 5]}
        # Make sure the appended data is not inplace modeified
        assert data2._split_idx_dict == data2_split_idx

    with subtests.test(mode="rename"):
        # Missing rename_dict
        pytest.raises(ValueError, data1.copy().append, data2, mode="rename")

        # Missing value for 'test'
        pytest.raises(KeyError, data1.copy().append, data2, mode="rename", rename_dict={"train": "new"})

        # Conflicting value for 'test'
        pytest.raises(ValueError, data1.copy().append, data2, mode="rename", rename_dict={"train": "a", "test": "test"})

        data = data1.copy()
        data.append(data2, mode="rename", rename_dict={"train": "new_train", "test": "new_test"})
        assert data._split_idx_dict == {"train": [0], "new_train": [3, 4], "test": [1, 2], "new_test": [5]}
        assert data2._split_idx_dict == data2_split_idx

    with subtests.test(mode="new_split"):
        # Missing new_split_name
        pytest.raises(ValueError, data1.copy().append, data2, mode="new_split")

        # Conflicting name "test"
        pytest.raises(ValueError, data1.copy().append, data2, mode="new_split", new_split_name="test")

        data = data1.copy()
        data.append(data2, mode="new_split", new_split_name="ref")
        assert data._split_idx_dict == {"train": [0], "test": [1, 2], "ref": [3, 4, 5]}
        assert data2._split_idx_dict == data2_split_idx

    with subtests.test(mode=None):
        data = data1.copy()
        data.append(data2, mode=None)
        assert data._split_idx_dict == {"train": [0], "test": [1, 2]}
        assert data2._split_idx_dict == data2_split_idx
        assert data2._split_idx_dict == data2_split_idx
        assert "batch" not in data.data.obs

    with subtests.test(label_batch=True):
        data = data1.copy()
        data.append(data2, mode=None, label_batch=True)
        assert data._split_idx_dict == {"train": [0], "test": [1, 2]}
        assert data2._split_idx_dict == data2_split_idx
        assert "batch" in data.data.obs
        assert data.data.obs["batch"].tolist() == [0, 0, 0, 1, 1, 1]

        data.append(data2, mode=None, label_batch=True)
        assert data.data.obs["batch"].tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
