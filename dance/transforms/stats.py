from pprint import pformat

import pandas as pd

from dance.registers import GENESTATS_FUNCS, register_genestats_func
from dance.transforms.base import BaseTransform
from dance.typing import List, Optional, Union
from dance.utils.wrappers import as_1d_array


class GeneStats(BaseTransform):
    """Gene statistics computation.

    Parameters
    ----------
    genestats_select
        List of names of the gene stats functions to use. If set to ``"all"`` (by default), then use all available gene
        stats functions.
    fill_na
        If not set (default), then do not fill nans. Otherwise, fill nans with the specified value.
    threshold
        Threshold value for filtering gene expression when computing stats, e.g., mean expression values.
    pseudo
        If set to ``True``, then add ``1`` to the numerator and denominator when computing the ratio (``alpha``) for
        which the gene expression values are above the specified ``threshold``.
    split_name
        Which split to compute the gene stats on.

    """

    _DISPLAY_ATTRS = ("genestats_select", "threshold", "pseudo", "split_name")

    def __init__(self, genestats_select: Union[str, List[str]] = "all", *, fill_na: Optional[float] = None,
                 threshold: float = 0, pseudo: bool = False, split_name: Optional[str] = "train",
                 channel: Optional[str] = None, channel_type: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)

        # Check genestats options
        if isinstance(genestats_select, str) and (genestats_select == "all"):
            self.genestats_select = list(GENESTATS_FUNCS)
        elif isinstance(genestats_select, list):
            invalid_options = [i for i in genestats_select if i not in GENESTATS_FUNCS]
            if invalid_options:
                raise ValueError(f"The following genestats selections are unavailable:\n{pformat(invalid_options)}\n"
                                 f"Currently supported genestats options are {pformat(list(GENESTATS_FUNCS))}")
            self.genestats_select = genestats_select

        # Set kwargs to be used by genestats functions
        self.func_kwargs = {
            "threshold": threshold,
            "pseudo": pseudo,
        }

        self.fill_na = fill_na
        self.split_name = split_name

        # Check expression layer option
        if (channel is not None) and (channel_type != "layers"):
            raise ValueError("Only the `layers` channels are available for selection other than the default `.X` "
                             "channel.\nPlease set `channel_type='layers'` to acknowledge this and resolve the error.")
        self.channel = channel
        self.channel_type = channel_type

    def __call__(self, data):
        exp = data.get_feature(return_type="numpy", split_name=self.split_name, channel=self.channel,
                               channel_type=self.channel_type)
        self.logger.info(f"Start computing gene stats: {self.genestats_select}")

        stats_dict = {}
        for name in self.genestats_select:
            func = GENESTATS_FUNCS[name]
            stats_dict[name] = func(exp, **self.func_kwargs)
        stats_df = pd.DataFrame(stats_dict, index=data.data.var_names)

        if self.fill_na is not None:
            stats_df = stats_df.fillna(self.fill_na)

        data.data.varm[self.out] = stats_df


@register_genestats_func("mu")
@as_1d_array
def genestats_mu(exp, threshold: float = 0, **kwargs):
    mask = (exp > threshold).astype(float)
    mu = (exp * mask).sum(0) / mask.sum(0)
    return mu


@register_genestats_func("alpha")
@as_1d_array
def genestats_alpha(exp, threshold: float = 0, pseudo: bool = False, **kwargs):
    mask = (exp > threshold).astype(float)
    count = mask.sum(0)
    total = exp.shape[0]

    if pseudo:
        count += 1
        total += 1

    alpha = count / total
    return alpha


@register_genestats_func("mean_all")
@as_1d_array
def genestats_mean_all(exp, **kwargs):
    return exp.mean(0)


@register_genestats_func("cov_all")
@as_1d_array
def genestats_cov_all(exp, **kwargs):
    return exp.std(0) / exp.mean(0)


@register_genestats_func("fano_all")
@as_1d_array
def genestats_fano_all(exp, **kwargs):
    return exp.var(0) / exp.mean(0)


@register_genestats_func("max_all")
@as_1d_array
def genestats_max_all(exp, **kwargs):
    return exp.max(0)


@register_genestats_func("std_all")
@as_1d_array
def genestats_std_all(exp, **kwargs):
    return exp.std(0)
