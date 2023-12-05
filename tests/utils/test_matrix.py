import itertools

import numpy as np
import scipy.stats

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


def test_pairwise_distance(subtests):
    mat = np.array([
        [0, 1, 2],
        [2, 2, 4],
        [5, 3, 5],
        [3, 2, 1],
        [5, 6, 3],
    ], dtype=np.float32)  # yapf: disable

    def compute_pairwise(ary, func):
        size = ary.shape[0]
        out = np.zeros((size, size), dtype=np.float32)
        for i, j in itertools.product(range(size), range(size)):
            if i <= j:
                out[i, j] = out[j, i] = func(ary[i], ary[j])
        return out

    with subtests.test("euclidean"):
        ans = compute_pairwise(mat, scipy.spatial.distance.euclidean)
        res = matrix.pairwise_distance(mat, 0)
        print(f"\n\n{ans=}\n{res=}")
        assert np.allclose(ans, res)

    with subtests.test("pearson"):
        ans = compute_pairwise(mat, lambda x, y: 1 - scipy.stats.pearsonr(x, y)[0])
        res = matrix.pairwise_distance(mat, 1)
        print(f"\n\n{ans=}\n{res=}")
        assert np.allclose(ans, res)

    with subtests.test("spearman"):
        ans = compute_pairwise(mat, lambda x, y: 1 - scipy.stats.spearmanr(x, y)[0])
        res = matrix.pairwise_distance(mat, 2)
        print(f"\n\n{ans=}\n{res=}")
        assert np.allclose(ans, res)
