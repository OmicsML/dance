import numpy as np
from anndata import AnnData

from dance import logger
from dance.exceptions import DevError
from dance.transforms.base import BaseTransform
from dance.typing import Dict, List, Literal, Optional, Union


class FilterGenesCommon(BaseTransform):
    """Filter genes by taking the common genes across batches or splits.

    Parameters
    ----------
    batch_key
        Which column in the ``.obs`` table to be used to distinguishing batches.
    split_keys
        A list of split names, e.g., 'train', to be used to find common gnees.

    Note
    ----
    One and only one of :attr:`batch_key` or :attr:`split_keys` can be specified.

    """

    _DISPLAY_ATTRS = ("batch_key", "split_keys")

    def __init__(self, batch_key: Optional[str] = None, split_keys: Optional[List[str]] = None, **kwargs):
        super().__init__(**kwargs)

        if (batch_key is not None) and (split_keys is not None):
            raise ValueError("Either batch_key or split_keys can be specified, but not both. "
                             f"Got {batch_key=!r}, {split_keys=!r}")
        elif (batch_key is None) and (split_keys is None):
            raise ValueError("Either one of batch_key or split_keys must be specified.")
        self.batch_key = batch_key
        self.split_keys = split_keys

    def _select_by_splits(self, data) -> Dict[Union[str, int], AnnData]:
        sliced_data_dict = {}
        for split_key in self.split_keys:
            idx = data.get_split_idx(split_key, error_on_miss=True)
            sliced_data_dict[split_key] = data.data[idx]
        return sliced_data_dict

    def _select_by_batch(self, data) -> Dict[Union[str, int], AnnData]:
        sliced_data_dict = {}
        for batch_id, group in data.data.obs.groupby(self.batch_key):
            sliced_data_dict[batch_id] = data.data[group.index]
        return sliced_data_dict

    def __call__(self, data):
        if self.batch_key is None:
            sliced_data_dict = self._select_by_splits(data)
        elif self.split_keys is None:
            sliced_data_dict = self._select_by_batch(data)
        else:
            raise DevError("Both batch_key and split_keys are not set. This should have been caught at init.")

        all_genes = data.data.var_names.tolist()
        sub_genes_list = []
        for name, sliced_data in sliced_data_dict.items():
            x = sliced_data.X
            abs_sum = np.array(np.abs(x).sum(0)).ravel()
            hits = np.where(abs_sum > 0)[0]
            sub_genes = [all_genes[i] for i in hits]
            sub_genes_list.append(sub_genes)
            logger.info(f"{len(sub_genes):,} genes found in {name!r}")

        common_genes = sorted(set.intersection(*map(set, sub_genes_list)))
        logger.info(f"Found {len(common_genes):,} common genes out of {len(all_genes):,} total genes.")
        data.data._inplace_subset_var(common_genes)


class FilterGenesMatch(BaseTransform):
    """Filter genes based on prefixes and suffixes.

    Parameters
    ----------
    prefixes
        List of prefixes to remove.
    suffixes
        List of suffixes to remove.

    """

    _DISPLAY_ATTRS = ("prefixes", "suffixes")

    def __init__(self, prefixes: Optional[List[str]] = None, suffixes: Optional[List[str]] = None,
                 case_sensitive: bool = False, **kwargs):
        super().__init__(**kwargs)

        self.prefixes = prefixes or []
        self.suffixes = suffixes or []
        self.case_sensitive = case_sensitive

        if case_sensitive:
            self.prefixes = [i.upper() for i in self.prefixes]
            self.suffixes = [i.upper() for i in self.suffixes]

    def __call__(self, data):
        indicator = np.zeros(data.shape[1], dtype=bool)

        for name, items in zip(["prefix", "suffix"], [self.prefixes, self.suffixes]):
            for item in items:
                ids = data.data.var_names.str
                if self.case_sensitive:
                    ids = ids.upper().str

                new_indicator = ids.startswith(item) if name == "prefix" else ids.endswith(item)
                logger.info(f"{new_indicator.sum()} number of genes will be removed due to {name} {item!r}")

                indicator = np.logical_or(indicator, new_indicator)

        logger.info(f"Removing {indicator.sum()} genes in total")
        logger.debug(f"Removing genes: {data.data.var_names[indicator]}")
        data.data._inplace_subset_var(data.data.var_names[~indicator])

        return data


class FilterGenesPercentile(BaseTransform):
    """Filter genes based on percentiles of the summarized gene expressions.

    Parameters
    ----------
    min_val
        Minimum percentile of the summarized expression value below which the genes will be discarded.
    max_val
        Maximum percentile of the summarized expression value above which the genes will be discarded.
    mode
        Summarization mode. Available options are ``[sum|cv]``. ``sum`` calculates the sum of expression values, ``cv``
        uses the coefficient of variation (std / mean).
    channel
        Which channel, more specificailly, ``layers``, to use. Use the default ``.X`` if not set. If ``channel`` is
        specified, then need to specify ``channel_type`` to be ``layers`` as well.
    channel_type
        Type of channels specified. Only allow ``None`` (the default setting) or ``layers`` (when ``channel`` is
        specified).

    """

    _DISPLAY_ATTRS = ("min_val", "max_val", "mode")
    _MODES = ["sum", "cv"]

    def __init__(self, min_val: Optional[float] = 1, max_val: Optional[float] = 99, mode: Literal["sum", "cv"] = "sum",
                 *, channel: Optional[str] = None, channel_type: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)

        if (channel is not None) and (channel_type != "layers"):
            raise ValueError(f"Only X layers is available for filtering genes, specified {channel_type=!r}")

        if mode not in self._MODES:
            raise ValueError(f"Unknown summarization mode {mode!r}, available options are {self._MODES}")

        self.min_val = min_val
        self.max_val = max_val
        self.mode = mode
        self.channel = channel
        self.channel_type = channel_type

    def __call__(self, data):
        x = data.get_feature(return_type="default", channel=self.channel, channel_type=self.channel_type)

        if self.mode == "sum":
            gene_summary = np.array(x.sum(0)).ravel()
        elif self.mode == "cv":
            gene_summary = np.nan_to_num(np.array(x.std(0) / x.mean(0)), posinf=0, neginf=0).ravel()
        else:
            raise DevError(f"{self.mode!r} not expected, please inform dev to fix this error.")

        percentile_lo = np.percentile(gene_summary, self.min_val)
        percentile_hi = np.percentile(gene_summary, self.max_val)
        mask = np.logical_and(gene_summary >= percentile_lo, gene_summary <= percentile_hi)
        self.logger.info(f"Filtering genes based on {self.mode} expression percentiles in layer {self.channel!r}")
        self.logger.info(f"{mask.size - mask.sum()} genes removed ({percentile_lo=:.2e}, {percentile_hi=:.2e})")

        data._data = data.data[:, mask].copy()
