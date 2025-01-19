import anndata
import mudata
import numpy as np
import pytest

from dance.data.base import Data
from dance.utils.wrappers import add_mod_and_transform


@pytest.fixture
def mock_mudata():
    """创建测试用的MuData对象."""
    adata1 = anndata.AnnData(X=np.array([[1, 2], [3, 4]]))
    adata2 = anndata.AnnData(X=np.array([[5, 6], [7, 8]]))
    return Data(data=mudata.MuData({'mod1': adata1, 'mod2': adata2}))


# 创建一个示例类来测试装饰器
@add_mod_and_transform
class SampleClass:
    """示例类，用于测试装饰器."""

    def __init__(self, x=10, **kwargs):
        self.x = x

    def __call__(self, data, *args, **kwargs):
        return data


def test_class_init_with_mod():
    """测试类初始化时带mod参数."""
    obj = SampleClass(x=10, mod="mod1")
    assert obj.x == 10
    assert obj.mod == "mod1"


def test_class_init_without_mod():
    """测试类初始化时不带mod参数."""
    obj = SampleClass(x=10)
    assert obj.x == 10
    assert obj.mod is None


def test_class_call_with_anndata():
    """测试使用AnnData对象调用."""
    obj = SampleClass(x=10)
    adata = Data(data=anndata.AnnData(X=np.array([[1, 2], [3, 4]])))
    result = obj(adata)
    assert result is adata


def test_class_call_with_mudata_and_mod(mock_mudata):
    """测试使用MuData对象和mod参数调用."""
    obj = SampleClass(x=10, mod="mod1")
    original_data = mock_mudata.data.mod["mod1"].copy()
    obj(mock_mudata)
    # 验证原始数据被正确修改
    assert np.array_equal(mock_mudata.data.mod["mod1"].X, original_data.X)


def test_class_call_with_mudata_without_mod(mock_mudata):
    """测试使用MuData对象但不带mod参数调用."""
    obj = SampleClass(x=10)
    result = obj(mock_mudata)
    assert result is mock_mudata


def test_class_call_with_mudata_invalid_mod(mock_mudata):
    """测试使用无效的mod参数."""
    obj = SampleClass(x=10, mod="invalid_mod")
    with pytest.raises(KeyError):
        obj(mock_mudata)


def test_decorator_preserves_metadata():
    """测试装饰器是否保留了原始类的元数据."""
    assert hasattr(SampleClass, 'add_mod_and_transform')
    assert SampleClass.__init__.__doc__ == SampleClass.__init__.__wrapped__.__doc__
    assert SampleClass.__call__.__doc__ == SampleClass.__call__.__wrapped__.__doc__


def test_class_call_with_additional_args(mock_mudata):
    """测试带额外参数调用."""
    obj = SampleClass(x=10, mod="mod1")
    original_data = mock_mudata.data.mod["mod1"].copy()
    obj(mock_mudata, extra_arg="test")
    # 验证原始数据被正确修改
    assert np.array_equal(mock_mudata.data.mod["mod1"].X, original_data.X)
