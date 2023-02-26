from functools import partial

import anndata as ad
import numpy as np
import pandas as pd

from dance import logger as native_logger
from dance.data import Data
from dance.transforms.base import BaseTransform
from dance.typing import Callable, Dict, List, Literal, Logger, Optional, Tuple, Union


class PseudoMixture(BaseTransform):

    _DISPLAY_ATTRS = ("n_pseudo", "nc_min", "nc_max", "ct_select")

    def __init__(
        self,
        *,
        n_pseudo: int = 1000,
        nc_min: int = 2,
        nc_max: int = 10,
        ct_select: Union[Literal["auto"], List[str]] = "auto",
        ct_key: str = "cellType",
        channel: Optional[str] = None,
        channel_type: Optional[str] = "X",
        random_state: Optional[int] = 0,
        prefix: str = "ps_mix_",
        in_split_name: str = "ref",
        out_split_name: Optional[str] = "pseudo",
        label_batch: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.n_pseudo = n_pseudo
        self.nc_min = nc_min
        self.nc_max = nc_max

        self.ct_select = ct_select
        self.ct_key = ct_key
        self.channel = channel
        self.channel_type = channel_type

        self.random_state = random_state
        self.prefix = prefix
        self.in_split_name = in_split_name
        self.out_split_name = out_split_name
        self.label_batch = label_batch

    @staticmethod
    def gen_mix(
        x: np.ndarray,
        annot: np.ndarray,
        nc_min: int = 2,
        nc_max: int = 10,
        clust_vr: str = "cellType",
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, Dict[str, int], Dict[str, float]]:
        rng = rng or np.random.default_rng()
        n_mix = rng.integers(nc_min, nc_max + 1)
        sample_inds = rng.choice(x.shape[0], size=n_mix, replace=False)

        mix_counts = x[sample_inds].sum(0)
        ct_counts_dict = dict(zip(*np.unique(annot[sample_inds], return_counts=True)))
        info_dict = {"cell_count": n_mix, "total_umi_count": mix_counts.sum()}

        return mix_counts, ct_counts_dict, info_dict

    def __call__(self, data):
        x = data.get_feature(split_name=self.in_split_name, channel=self.channel, channel_type=self.channel_type,
                             return_type="numpy")
        annot = data.get_feature(split_name=self.in_split_name, channel=self.ct_key, channel_type="obs",
                                 return_type="numpy")

        rng = np.random.default_rng(self.random_state)
        bouned_gen_mix = partial(self.gen_mix, nc_min=self.nc_min, nc_max=self.nc_max, clust_vr=self.ct_key, rng=rng)

        mix_x = np.zeros((self.n_pseudo, x.shape[1]), dtype=np.float32)
        ct_counts_dict_list, ps_info_dict_list = [], []
        for i in range(self.n_pseudo):
            mix_x[i], ct_counts_dict, info_dict = bouned_gen_mix(x, annot)
            ct_counts_dict_list.append(ct_counts_dict)
            ps_info_dict_list.append(info_dict)

        ct_select = get_cell_types(self.ct_select, annot)
        index_list = [f"{self.prefix}{i}" for i in range(self.n_pseudo)]
        ct_counts_df = pd.DataFrame(ct_counts_dict_list, columns=ct_select, index=index_list).fillna(0)
        ct_portion_df = ct_counts_df.div(ct_counts_df.sum(axis=1), axis=0)  # normalize to portions
        obs = pd.DataFrame(ps_info_dict_list, index=index_list)

        pseudo_data = Data(ad.AnnData(mix_x, obs=obs, var=data.data.var, obsm={"cell_type_portion": ct_portion_df}))
        data.append(pseudo_data, join="outer", mode="new_split", new_split_name=self.out_split_name,
                    label_batch=self.label_batch)


class CellTopicProfile(BaseTransform):

    _DISPLAY_ATTRS = ("ct_select", "ct_key", "split_name", "method")

    def __init__(
        self,
        *,
        ct_select: Union[Literal["auto"], List[str]] = "auto",
        ct_key: str = "cellType",
        batch_key: Optional[str] = None,
        split_name: Optional[str] = None,
        channel: Optional[str] = None,
        channel_type: str = "X",
        method: Literal["median", "mean"] = "median",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.ct_select = ct_select
        self.ct_key = ct_key
        self.batch_key = batch_key
        self.split_name = split_name

        self.channel = channel
        self.channel_type = channel_type
        self.method = method

    def __call__(self, data):
        x = data.get_feature(split_name=self.split_name, channel=self.channel, channel_type=self.channel_type,
                             return_type="numpy")
        annot = data.get_feature(split_name=self.split_name, channel=self.ct_key, channel_type="obs",
                                 return_type="numpy")
        if self.batch_key is None:
            batch_index = None
        else:
            batch_index = data.get_feature(split_name=self.split_name, channel=self.batch_key, channel_type="obs",
                                           return_type="numpy")

        ct_select = get_cell_types(self.ct_select, annot)
        ct_profile = get_ct_profile(x, annot, batch_index=batch_index, ct_select=ct_select, method=self.method,
                                    logger=self.logger)
        ct_profile_df = pd.DataFrame(ct_profile, index=data.data.var_names, columns=ct_select)

        data.data.varm[self.out] = ct_profile_df


def get_cell_types(ct_select: Union[Literal["auto"], List[str]], annot: np.ndarray) -> List[str]:
    all_cts = sorted(np.unique(annot))
    if ct_select == "auto":
        ct_select = all_cts
    elif len(missed := sorted(set(ct_select) - set(all_cts))) > 0:
        raise ValueError(f"Unknown cell types selected: {missed}. Available options are: {all_cts}")
    return ct_select


def get_agg_func(name: str, *, default: Optional[str] = None) -> Callable[[np.ndarray], np.ndarray]:
    if name == "default":
        if default is None:
            raise ValueError("Aggregation function name set to 'default' but default option not set")
        name = default

    if name == "median":
        agg_func = partial(np.median, axis=0)
    elif name == "mean":
        agg_func = partial(np.mean, axis=0)
    else:
        raise ValueError(f"Unknown aggregation method {name!r}. Available options are: 'median', 'mena'")

    return agg_func


def get_ct_profile(
    x: np.ndarray,
    annot: np.ndarray,
    *,
    batch_index: Optional[np.ndarray] = None,
    ct_select: Union[Literal["auto"], List[str]] = "auto",
    method: Literal["median", "mean"] = "mean",
    logger: Optional[Logger] = None,
) -> np.ndarray:
    logger = logger or native_logger
    ct_select = get_cell_types(ct_select, annot)
    agg_func = get_agg_func(method, default="mean")
    if batch_index is None:
        batch_index = np.zeros(x.shape[0], dtype=int)

    # Aggregate profile for each selected cell types
    logger.info(f"Generating cell-type profiles ({method!r} aggregation) for {ct_select}")
    ct_profile = np.zeros((x.shape[1], len(ct_select)), dtype=np.float32)  # gene x cell

    for i, ct in enumerate(ct_select):
        ct_index = np.where(annot == ct)[0]
        logger.info(f"Aggregating {ct!r} profiles over {ct_index.size:,} samples")

        # Get features within a cell type
        sub_batch_index = batch_index[ct_index]
        batches = np.unique(sub_batch_index)

        # Aggregate cell type profile for each batch
        sub_ct_profile = np.zeros((batches.size, x.shape[1]), dtype=np.float32)  # cell x gene
        sub_ct_mean_lib_sizes = np.zeros(batches.size, dtype=np.float32)
        for j, batch_id in enumerate(batches):
            idx = np.where(sub_batch_index == batch_id)[0]
            sub_ct_profile[j] = agg_func(x[ct_index][idx])
            sub_ct_mean_lib_sizes[j] = sub_ct_profile[j].sum()
            sub_ct_profile[j] /= sub_ct_mean_lib_sizes[j]
            logger.info(f"Number of {ct!r} cells in batch {batch_id!r}: {idx.size:,}")

        # Aggregate cell type profile over batches
        ct_profile[:, i] = agg_func(sub_ct_profile) * agg_func(sub_ct_mean_lib_sizes)

    logger.info("Cell-type profile generated")

    return ct_profile
