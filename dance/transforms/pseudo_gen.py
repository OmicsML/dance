from functools import partial

import numpy as np

from dance import logger as native_logger
from dance.transforms.base import BaseTransform
from dance.typing import List, Literal, Logger, Optional, Union


class CellTopicProfile(BaseTransform):

    _DISPLAY_ATTRS = ("ct_select", "split_name")

    def __init__(
        self,
        ct_select: Union[Literal["auto"], List[str]] = "auto",
        *,
        ct_key: str = "cellType",
        split_name: Optional[str] = None,
        channel: Optional[str] = None,
        channel_type: str = "X",
        method: Literal["median", "mean"] = "median",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.ct_select = ct_select
        self.ct_key = ct_key
        self.channel = channel
        self.channel_type = channel_type

        self._get_agg_func(method)  # check if aggregation method is valid
        self.method

    def __call__(self, data):
        x = data.get_feature(split_name=self.split_name, channel=self.feature_channel,
                             channel_type=self.feature_channel_type, return_type="numpy")
        annot = data.get_feature(split_name=self.split_name, channel=self.ct_channel, channel_type=self.ct_channel_type,
                                 return_type="numpy")

        ct_profile = get_ct_profile(x, annot, self.ct_select, self.method, self.logger)

        data.data.varm[self.out] = ct_profile


def get_ct_profile(
    x: np.ndarray,
    annot: np.ndarray,
    /,
    ct_select: Union[Literal["auto"], List[str]] = "auto",
    method: Literal["median", "mean"] = "median",
    logger: Optional[Logger] = None,
) -> np.ndarray:
    logger = logger or native_logger

    # Get aggregation function
    if method == "median":
        agg_func = partial(np.median, axis=0)
    elif method == "mean":
        agg_func = partial(np.mean, axis=0)
    else:
        raise ValueError(f"Unknown aggregation method {method!r}. Available options are: 'median', 'mena'")

    # Get cell type selections
    all_cts = sorted(np.unique(annot))
    if ct_select == "auto":
        ct_select = all_cts
    elif len(missed := sorted(set(ct_select) - set(all_cts))) > 0:
        raise ValueError(f"Unknown cell types selected: {missed}. Available options are: {all_cts}")

    # Aggregate profile for each selected cell types
    logger.info(f"Generating cell-type profiles ({method!r} aggregation) for {ct_select}")
    ct_profile = np.zeros((len(ct_select), x.shape[1]), dtype=np.float32)
    for i, ct in enumerate(ct_select):
        ct_index = np.where(annot == ct)[0]
        logger.info(f"Aggregating {ct!r} profiles over {ct_index.size:,} samples")
        ct_profile[i] = agg_func(x[ct_index])
    logger.info("Cell-type profile generated")

    return ct_profile
