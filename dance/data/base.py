import torch
from anndata import AnnData

from dance.typing import FeatType, List, Optional, ReturnedFeat, Sequence, Tuple, Union


class Data:
    """Base data object."""

    def __init__(self, x: Optional[AnnData] = None, y: Optional[AnnData] = None):
        """Initialize data object.

        Parameters
        ----------
        x : AnnData
            Cell features.
        y : AnnData
            Cell labels.

        """
        self._x = x or AnnData()
        self._y = y

        self._split_idx_dict = {}

    @property
    def x(self) -> AnnData:
        return self._x

    @property
    def y(self) -> Optional[AnnData]:
        return self._y

    @property
    def num_cells(self) -> int:
        return self.x.shape[0]

    @property
    def num_features(self) -> int:
        return self.x.shape[1]

    @property
    def cells(self) -> List[str]:
        return self.x.obs.index.tolist()

    def set_split_idx(self, split_name: str, split_idx: Sequence[Union[int, str]]):
        """Set cell indices for a particular split.

        Parameters
        ----------
        split_name
            Name of the split to set.
        split_idx
            Indices of the cells to be used in this split.

        """
        self._split_idx_dict[split_name] = split_idx

    def get_split_idx(self, split_name: str):
        """Obtain cell indices for a particular split.

        Parameters
        ----------
        split_name : str
            Name of the split to retrieve.

        """
        return self._split_idx_dict[split_name]

    def _get_feat(self, feat_name: str, split_name: Optional[str], return_type: FeatType = "numpy"):
        if split_name is None:
            feat = getattr(self, feat_name)
        elif split_name in self._split_idx_dict:
            idx = self.get_split_idx(split_name)
            feat = getattr(self, feat_name)[idx]
        else:
            raise KeyError(f"Unknown split {split_name!r}, available options are {list(split_name)}")

        if return_type != "anndata":
            feat = feat.X

            try:  # convert sparse array to dense array
                feat = feat.toarray()
            except AttributeError:
                pass

            if return_type == "torch":
                feat = torch.from_numpy(feat)
            elif return_type != "numpy":
                raise ValueError(f"Unknown return_type {return_type!r}")

        return feat

    def get_x(self, split_name: Optional[str] = None, return_type: FeatType = "numpy") -> ReturnedFeat:
        """Retrieve cell features from a particular split."""
        return self._get_feat("x", split_name, return_type)

    def get_y(self, split_name: Optional[str] = None, return_type: FeatType = "numpy") -> ReturnedFeat:
        """Retrieve cell labels from a particular split."""
        return self._get_feat("y", split_name, return_type)

    def get_x_y(self, split_name: Optional[str] = None,
                return_type: FeatType = "numpy") -> Tuple[ReturnedFeat, ReturnedFeat]:
        """Retrieve cell features and labels from a particular split.

        Parameters
        ----------
        split_name
            Name of the split to retrieve. If not set, return all.
        return_type
            How should the features be returned. **numpy**: return as a numpy array; **torch**: return as a torch
            tensor; **anndata**: return as an anndata object.

        """
        x = self.get_x(split_name, return_type)
        y = self.get_y(split_name, return_type)
        return x, y

    def get_train_data(self, return_type: FeatType = "numpy") -> Tuple[ReturnedFeat, ReturnedFeat]:
        """Retrieve cell features and labels from the 'train' split."""
        return self.get_x_y("train", return_type)

    def get_val_data(self, return_type: FeatType = "numpy") -> Tuple[ReturnedFeat, ReturnedFeat]:
        """Retrieve cell features and labels from the 'val' split."""
        return self.get_x_y("val", return_type)

    def get_test_data(self, return_type: FeatType = "numpy") -> Tuple[ReturnedFeat, ReturnedFeat]:
        """Retrieve cell features and labels from the 'test' split."""
        return self.get_x_y("test", return_type)
