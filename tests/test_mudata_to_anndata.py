import anndata
import mudata
import numpy as np
import pytest

from dance.data.base import Data
from dance.utils.wrappers import add_mod_and_transform


@pytest.fixture
def mock_mudata():
    """Create a MuData object for testing."""
    adata1 = anndata.AnnData(X=np.array([[1, 2], [3, 4]]))
    adata2 = anndata.AnnData(X=np.array([[5, 6], [7, 8]]))
    return Data(data=mudata.MuData({'mod1': adata1, 'mod2': adata2}))


@add_mod_and_transform
class SampleClass:
    """Example class for testing the decorator."""

    def __init__(self, x=10, **kwargs):
        self.x = x

    def __call__(self, data, *args, **kwargs):
        # Multiply the data by self.x
        if isinstance(data.data, anndata.AnnData):
            data.data.X = data.data.X * self.x
        return data


def test_class_init_with_mod():
    """Test class initialization with mod parameter."""
    obj = SampleClass(x=10, mod="mod1")
    assert obj.x == 10
    assert obj.mod == "mod1"


def test_class_init_without_mod():
    """Test class initialization without mod parameter."""
    obj = SampleClass(x=10)
    assert obj.x == 10
    assert obj.mod is None


def test_class_call_with_anndata():
    """Test calling with AnnData object."""
    obj = SampleClass(x=3)
    original_data = np.array([[1, 2], [3, 4]])
    adata = Data(data=anndata.AnnData(X=original_data.copy()))
    result = obj(adata)
    # Verify data is multiplied by x=3
    assert np.array_equal(result.data.X, original_data * 3)


def test_class_call_with_mudata_and_mod(mock_mudata):
    """Test calling with MuData object and mod parameter to verify that only the
    specified modality is modified."""
    obj = SampleClass(x=2, mod="mod1")
    # Store original data for both modalities
    original_mod1 = mock_mudata.data.mod["mod1"].X.copy()
    original_mod2 = mock_mudata.data.mod["mod2"].X.copy()

    obj(mock_mudata)

    # Verify mod1 data is multiplied by x=2
    assert np.array_equal(mock_mudata.data.mod["mod1"].X, original_mod1 * 2)
    # Verify mod2 data remains unchanged
    assert np.array_equal(mock_mudata.data.mod["mod2"].X, original_mod2)


def test_class_call_with_mudata_without_mod(mock_mudata):
    """Test calling with MuData object but without mod parameter."""
    obj = SampleClass(x=10)
    result = obj(mock_mudata)
    assert result is mock_mudata


def test_class_call_with_mudata_invalid_mod(mock_mudata):
    """Test using invalid mod parameter."""
    obj = SampleClass(x=10, mod="invalid_mod")
    with pytest.raises(KeyError):
        obj(mock_mudata)


def test_decorator_preserves_metadata():
    """Test if the decorator preserves the original class metadata."""
    assert hasattr(SampleClass, 'add_mod_and_transform')
    assert SampleClass.__init__.__doc__ == SampleClass.__init__.__wrapped__.__doc__
    assert SampleClass.__call__.__doc__ == SampleClass.__call__.__wrapped__.__doc__


def test_class_call_with_additional_args(mock_mudata):
    """Test calling with additional arguments."""
    obj = SampleClass(x=10, mod="mod1")
    original_data = mock_mudata.data.mod["mod1"].copy()
    obj(mock_mudata, extra_arg="test")
    # Verify original data is modified correctly
    assert np.array_equal(mock_mudata.data.mod["mod1"].X, original_data.X * 10)
