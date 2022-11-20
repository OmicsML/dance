import numpy as np

from dance.utils import matrix


def test_normalize(subtests):
    mat = np.array([[1, 1], [4, 4]])

    with subtests.test("normalize"):
        assert matrix.normalize(mat, mode="normalize", axis=0).tolist() == [[0.2, 0.2], [0.8, 0.8]]
        assert matrix.normalize(mat, mode="normalize", axis=1).tolist() == [[0.5, 0.5], [0.5, 0.5]]

    with subtests.test("standardize"):
        assert matrix.normalize(mat, mode="standardize", axis=0).tolist() == [[-1, -1], [1, 1]]
        assert matrix.normalize(mat, mode="standardize", axis=1).tolist() == [[0, 0], [0, 0]]

    with subtests.test("minmax"):
        assert matrix.normalize(mat, mode="minmax", axis=0).tolist() == [[0, 0], [1, 1]]
        assert matrix.normalize(mat, mode="minmax", axis=1).tolist() == [[0, 0], [0, 0]]

    with subtests.test("l2"):
        mat_norm0 = (mat / np.sqrt((mat**2).sum(0))).tolist()
        assert matrix.normalize(mat, mode="l2", axis=0).tolist() == mat_norm0

        mat_norm1 = (mat / np.sqrt((mat**2).sum(1, keepdims=True))).tolist()
        assert matrix.normalize(mat, mode="l2", axis=1).tolist() == mat_norm1
