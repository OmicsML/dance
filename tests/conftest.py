import numpy as np
import pytest


@pytest.fixture
def assert_ary_isclose():
    """Assert two numpy arrays are elementwise close based on numpy.isclose."""

    def func(x, y, /, *, rtol=1e-5, atol=1e-8, equal_nan=False):
        res = np.isclose(x, y, rtol=rtol, atol=atol, equal_nan=equal_nan)
        assert res.all(), (f"Comparison between two arrays failed isclose assertion:\n\n"
                           f"Array1 = {x}\n\nArray2 = {y}\n\n{(~res).sum()} out of {res.size} entries "
                           f"({(~res).sum() / res.size:.2%}) failed isclose assertion:\n{np.where(~res)}")

    return func
