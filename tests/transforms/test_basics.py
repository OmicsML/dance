from dance.transforms import CellPCA, WeightedFeaturePCA
from dance.transforms.graph import DSTGraph, NeighborGraph, PCACellFeatureGraph, SpaGCNGraph


def test_reprs(subtests):
    with subtests.test("CellPCA"):
        t = CellPCA(n_components=100)
        assert repr(t) == "CellPCA(n_components=100)"

    with subtests.test("WeightedFeaturePCA"):
        t = WeightedFeaturePCA(n_components=100, split_name="train")
        assert repr(t) == ("WeightedFeaturePCA(n_components=100, split_name='train', "
                           "feat_norm_mode=None, feat_norm_axis=0)")

    with subtests.test("NeighborGraph"):
        t = NeighborGraph(n_neighbors=10, n_pcs=None, knn=True, random_state=0, method="umap", metric="euclidean")
        assert repr(t) == ("NeighborGraph(n_neighbors=10, n_pcs=None, knn=True, random_state=0, method='umap', "
                           "metric='euclidean')")

    with subtests.test("PCACellFeatureGraph"):
        t = PCACellFeatureGraph(n_components=100, split_name="train")
        assert repr(t) == "PCACellFeatureGraph(n_components=100, split_name='train')"

    with subtests.test("DSTGraph"):
        t = DSTGraph(k_filter=100, num_cc=10, ref_split="train", inf_split="test")
        assert repr(t) == "DSTGraph(k_filter=100, num_cc=10, ref_split='train', inf_split='test')"

    with subtests.test("SpaGCNGraph"):
        t = SpaGCNGraph(alpha=1, beta=2)
        assert repr(t) == "SpaGCNGraph(alpha=1, beta=2)"
