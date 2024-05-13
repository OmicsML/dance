import warnings
from abc import ABC
from typing import get_args

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import rankdata
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

from dance import logger as default_logger
from dance.data.base import Data
from dance.exceptions import DevError
from dance.registry import register_preprocessor
from dance.transforms.base import BaseTransform
from dance.transforms.interface import AnnDataTransform
from dance.typing import Dict, GeneSummaryMode, List, Literal, Logger, Optional, Tuple, Union
from dance.utils import default
from dance.utils.status import deprecated


def get_count(count_or_ratio: Optional[Union[float, int]], total: int) -> Optional[int]:
    """Get the count from a count or ratio.

    Parameters
    ----------
    count_or_ratio
        Either a count or a ratio. If None, then return None.
    total
        Total number.

    """
    if count_or_ratio is None:
        return None
    elif isinstance(count_or_ratio, float):
        if count_or_ratio > 1.:
            raise ValueError(f"{count_or_ratio=} is greater than 1. Ratio cannot be greater than 1.")
        return int(count_or_ratio * total)
    elif isinstance(count_or_ratio, int):
        if count_or_ratio > total:
            raise ValueError(f"{count_or_ratio=} is greater than {total=}")
        return count_or_ratio
    else:
        raise TypeError(f"count_or_ratio must be either float or int, got {type(count_or_ratio)}")


@register_preprocessor("filter")
class FilterScanpy(BaseTransform):
    """Scanpy filtering transformation with additional options."""

    _FILTER_TARGET: Optional[Literal["cells", "genes"]] = None

    def __init__(
        self,
        min_counts: Optional[Union[float, int]] = None,
        min_genes_or_cells: Optional[Union[float, int]] = None,
        max_counts: Optional[Union[float, int]] = None,
        max_genes_or_cells: Optional[Union[float, int]] = None,
        split_name: Optional[str] = None,
        channel: Optional[str] = None,
        channel_type: Optional[str] = "X",
        key_n_counts: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.min_counts = min_counts
        self.min_genes_or_cells = min_genes_or_cells
        self.max_counts = max_counts
        self.max_genes_or_cells = max_genes_or_cells
        self.key_n_counts = key_n_counts
        self.split_name = split_name
        self.channel = channel
        self.channel_type = channel_type

        if self._FILTER_TARGET is None:
            raise NotImplementedError("Use FilterCellsScanpy or FilterGenesScanpy instead")
        elif self._FILTER_TARGET == "cells":
            self._subsetting_func_name = "_inplace_subset_obs"
            self._filter_func = sc.pp.filter_cells
            self.min_genes = min_genes_or_cells
            self.max_genes = max_genes_or_cells
        elif self._FILTER_TARGET == "genes":
            self._subsetting_func_name = "_inplace_subset_var"
            self._filter_func = sc.pp.filter_genes
            self.min_cells = min_genes_or_cells
            self.max_cells = max_genes_or_cells
        else:
            raise ValueError(f"Unknown filter target {self._FILTER_TARGET!r}")

    def __call__(self, data):
        x = data.get_feature(return_type="numpy", split_name=self.split_name, channel=self.channel,
                             channel_type=self.channel_type)
        total_cells, total_features = x.shape

        min_counts = self.min_counts
        max_counts = self.max_counts

        # Determine whether we are dealing with cells or genes
        basis = total_cells if self._FILTER_TARGET == "genes" else total_features
        other_name = "cells" if self._FILTER_TARGET == "genes" else "genes"
        opts = {
            "min_counts": min_counts,
            "max_counts": max_counts,
            f"min_{other_name}": get_count(self.min_genes_or_cells, basis),
            f"max_{other_name}": get_count(self.max_genes_or_cells, basis),
        }
        subset_ind, _ = self._filter_func(x, inplace=False, **opts)
        if self.key_n_counts is not None:
            self.logger.warning(f"{self.key_n_counts} will be added to the data")
            if self._FILTER_TARGET == "genes":
                if min_counts is not None or max_counts is not None:
                    n_counts = np.sum(x, axis=0)
                    data.data.var[self.key_n_counts] = n_counts
                else:
                    n_cells = np.sum(x > 0, axis=0)
                    data.data.var[self.key_n_counts] = n_cells
            else:
                if min_counts is not None or max_counts is not None:
                    n_counts = np.sum(x, axis=1)
                    data.data.obs[self.key_n_counts] = n_counts
                else:
                    n_genes = np.sum(x > 0, axis=1)
                    data.data.obs[self.key_n_counts] = n_genes

        if not subset_ind.all():
            subset_func = getattr(data.data, self._subsetting_func_name)
            self.logger.info(f"Subsetting {self._FILTER_TARGET} ({~subset_ind.sum():,} removed) due to {self}")
            subset_func(subset_ind)

    def prepCounts(self, x):
        if (isinstance(self.min_counts, float) and 0 < self.min_counts < 1) or (isinstance(self.max_counts, float)
                                                                                and 0 < self.max_counts < 1):
            if self._FILTER_TARGET == "genes":
                n_counts = np.sum(x, axis=0)
            elif self._FILTER_TARGET == "cells":
                n_counts = np.sum(x, axis=1)
            if isinstance(self.min_counts, float) and 0 <= self.min_counts <= 1:
                min_counts = np.percentile(n_counts, min_counts)
            else:
                max_counts = np.percentile(n_counts, max_counts)
            return min_counts, max_counts


@register_preprocessor("filter", "cell")
class FilterCellsScanpy(FilterScanpy):
    """Scanpy filtering cell transformation with additional options.

    Allow passing gene counts as ratio

    Parameters
    ----------
    min_counts
        Minimum number of counts required for a cell to be kept.
    min_genes
        Minimum number (or ratio) of genes required for a cell to be kept.
    max_counts
        Maximum number of counts required for a cell to be kept.
    max_genes
        Maximum number (or ratio) of genes required for a cell to be kept.
    split_name
        Which split to be used for filtering.
    channel
        Channel to be used for filtering.
    channel_type
        Channel type to be used for filtering.
    key_n_counts
        The location to add n_counts or n_cells/genes. If it is None, it will not be added.

    """

    _DISPLAY_ATTRS = ("min_counts", "min_genes", "max_counts", "max_genes", "split_name")
    _FILTER_TARGET = "cells"

    def __init__(
        self,
        min_counts: Optional[Union[float, int]] = None,
        min_genes: Optional[Union[float, int]] = None,
        max_counts: Optional[Union[float, int]] = None,
        max_genes: Optional[Union[float, int]] = None,
        split_name: Optional[str] = None,
        channel: Optional[str] = None,
        channel_type: Optional[str] = "X",
        key_n_counts: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            min_counts=min_counts,
            min_genes_or_cells=min_genes,
            max_counts=max_counts,
            max_genes_or_cells=max_genes,
            split_name=split_name,
            channel=channel,
            channel_type=channel_type,
            key_n_counts=key_n_counts,
            **kwargs,
        )


@register_preprocessor("filter", "gene")
class FilterGenesScanpy(FilterScanpy):
    """Scanpy filtering gene transformation with additional options.

    Parameters
    ----------
    min_counts
        Minimum number of counts required for a gene to be kept.
    min_cells
        Minimum number (or ratio) of cells required for a gene to be kept.
    max_counts
        Maximum number of counts required for a gene to be kept.
    max_cells
        Maximum number (or ratio) of cells required for a gene to be kept.
    split_name
        Which split to be used for filtering.
    channel
        Channel to be used for filtering.
    channel_type
        Channel type to be used for filtering.
    key_n_counts
        The location to add n_counts or n_cells/genes. If it is None, it will not be added.

    """

    _DISPLAY_ATTRS = ("min_counts", "min_cells", "max_counts", "max_cells", "split_name")
    _FILTER_TARGET = "genes"

    def __init__(
        self,
        min_counts: Optional[Union[float, int]] = None,
        min_cells: Optional[Union[float, int]] = None,
        max_counts: Optional[Union[float, int]] = None,
        max_cells: Optional[Union[float, int]] = None,
        split_name: Optional[str] = None,
        channel: Optional[str] = None,
        channel_type: Optional[str] = "X",
        key_n_counts: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(min_counts=min_counts, min_genes_or_cells=min_cells, max_counts=max_counts,
                         max_genes_or_cells=max_cells, split_name=split_name, channel=channel,
                         channel_type=channel_type, key_n_counts=key_n_counts, **kwargs)


@register_preprocessor("filter", "gene")
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

    def _select_by_splits(self, data) -> Dict[Union[str, int], ad.AnnData]:
        sliced_data_dict = {}
        for split_key in self.split_keys:
            idx = data.get_split_idx(split_key, error_on_miss=True)
            sliced_data_dict[split_key] = data.data[idx]
        return sliced_data_dict

    def _select_by_batch(self, data) -> Dict[Union[str, int], ad.AnnData]:
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
            self.logger.info(f"{len(sub_genes):,} genes found in {name!r}")

        common_genes = sorted(set.intersection(*map(set, sub_genes_list)))
        self.logger.info(f"Found {len(common_genes):,} common genes out of {len(all_genes):,} total genes.")
        data.data._inplace_subset_var(common_genes)


@register_preprocessor("filter", "gene")
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

    def __init__(
        self,
        prefixes: Optional[List[str]] = None,
        suffixes: Optional[List[str]] = None,
        case_sensitive: bool = False,
        **kwargs,
    ):
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
                self.logger.info(f"{new_indicator.sum()} number of genes will be removed due to {name} {item!r}")

                indicator = np.logical_or(indicator, new_indicator)

        self.logger.info(f"Removing {indicator.sum()} genes in total")
        self.logger.debug(f"Removing genes: {data.data.var_names[indicator]}")
        data.data._inplace_subset_var(data.data.var_names[~indicator])

        return data


class FilterGenes(BaseTransform, ABC):
    """Filter genes based on the summarized gene expressions."""

    def __init__(
        self,
        *,
        mode: GeneSummaryMode = "sum",
        channel: Optional[str] = None,
        channel_type: Optional[str] = None,
        whitelist_indicators: Optional[Union[str, List[str]]] = None,
        add_n_counts=True,
        add_n_cells=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if (channel is not None) and (channel_type != "layers"):
            raise ValueError(f"Only X layers is available for filtering genes, specified {channel_type=!r}")

        if mode not in (all_modes := sorted(get_args(GeneSummaryMode))):
            raise ValueError(f"Unknown summarization mode {mode!r}, available options are {all_modes}")

        self.mode = mode
        self.channel = channel
        self.channel_type = channel_type
        self.whitelist_indicators = whitelist_indicators
        self.add_n_counts = add_n_counts
        self.add_n_cells = add_n_cells

    def _get_preserve_mask(self, gene_summary: np.ndarray) -> np.ndarray:
        """Select gene to be preserved and return as a mask."""
        ...

    def __call__(self, data):
        x = data.get_feature(return_type="numpy", channel=self.channel, channel_type=self.channel_type)
        if self.add_n_counts:
            self.logger.warning(f"n_counts will be added to the var of data")
            n_counts = np.sum(x, axis=0)
            data.data.var["n_counts"] = n_counts
        if self.add_n_cells:
            self.logger.warning(f"n_cells will be added to the var of data")
            n_cells = np.sum(x > 0, axis=0)
            data.data.var["n_cells"] = n_cells
        # Compute gene summary stats for filtering
        if self.mode == "sum":
            gene_summary = np.array(x.sum(0)).ravel()
        elif self.mode == "var":
            x_squared = x.power(2) if isinstance(x, sp.spmatrix) else x**2
            gene_summary = np.array(x_squared.mean(0) - np.square(x.mean(0))).ravel()
        elif self.mode == "cv":
            gene_summary = np.nan_to_num(np.array(x.std(0) / x.mean(0)), posinf=0, neginf=0).ravel()
        elif self.mode == "rv":
            gene_summary = np.nan_to_num(np.array(x.var(0) / x.mean(0)), posinf=0, neginf=0).ravel()
        else:
            raise DevError(f"{self.mode!r} not expected, please inform dev to fix this error.")

        self.logger.info(f"Filtering genes based on {self.mode} expression percentiles in layer {self.channel!r}")
        mask = self._get_preserve_mask(gene_summary)
        selected_genes = sorted(data.data.var_names[mask])

        # Get whitelist genes to be excluded from the filtering process
        whitelist_gene_set = set()
        if self.whitelist_indicators is not None:
            columns = self.whitelist_indicators
            columns = [columns] if isinstance(columns, str) else columns
            indicators = data.data.var[columns]
            # Genes that satisfy any one of the whitelist conditions will be selected as whitelist genes
            whitelist_gene_set.update(indicators[indicators.max(1)].index.tolist())

        # Exclude whitelisted genes
        if len(whitelist_gene_set) > 0:
            orig_num_selected = len(selected_genes)
            selected_genes = sorted(set(selected_genes) | whitelist_gene_set)
            num_added = len(selected_genes) - orig_num_selected
            self.logger.info(f"{num_added:,} genes originally unselected are being added due to whitelist")

        # Update data
        self.logger.info(f"{data.shape[1] - len(selected_genes):,} genes removed")
        data.data._inplace_subset_var(selected_genes)


@register_preprocessor("filter", "gene")
class FilterGenesPercentile(FilterGenes):
    """Filter genes based on percentiles of the summarized gene expressions.

    Parameters
    ----------
    min_val
        Minimum percentile of the summarized expression value below which the genes will be discarded.
    max_val
        Maximum percentile of the summarized expression value above which the genes will be discarded.
    mode
        Summarization mode. Available options are ``[sum|var|cv|rv]``. ``sum`` calculates the sum of expression values,
        ``var`` calculates the variance of the expression values, ``cv`` uses the coefficient of variation (std / mean
        ), and ``rv`` uses the relative variance (var / mean).
    channel
        Which channel, more specificailly, ``layers``, to use. Use the default ``.X`` if not set. If ``channel`` is
        specified, then need to specify ``channel_type`` to be ``layers`` as well.
    channel_type
        Type of channels specified. Only allow ``None`` (the default setting) or ``layers`` (when ``channel`` is
        specified).
    whitelist_indicators
        A list of (or a single) :obj:`.var` columns that indicates the genes to be excluded from the filtering process.
        Note that these genes will still be used in the summary stats computation, and thus will still contribute to the
        threshold percentile. If not set, then no genes will be excluded from the filtering process.
    add_n_counts
        Whether to add n_counts
    add_n_cells
        Whether to add n_cells

    """

    _DISPLAY_ATTRS = ("min_val", "max_val", "mode")

    def __init__(
        self,
        min_val: Optional[float] = 1,
        max_val: Optional[float] = 99,
        *,
        mode: GeneSummaryMode = "sum",
        channel: Optional[str] = None,
        channel_type: Optional[str] = None,
        whitelist_indicators: Optional[Union[str, List[str]]] = None,
        add_n_counts=True,
        add_n_cells=True,
        **kwargs,
    ):
        super().__init__(
            mode=mode,
            channel=channel,
            channel_type=channel_type,
            whitelist_indicators=whitelist_indicators,
            add_n_counts=add_n_counts,
            add_n_cells=add_n_cells,
            **kwargs,
        )
        self.min_val = min_val
        self.max_val = max_val

    def _get_preserve_mask(self, gene_summary):
        percentile_lo = np.percentile(gene_summary, self.min_val)
        percentile_hi = np.percentile(gene_summary, self.max_val)
        return np.logical_and(gene_summary >= percentile_lo, gene_summary <= percentile_hi)


@register_preprocessor("filter", "gene")
class FilterGenesTopK(FilterGenes):
    """Select top/bottom genes based on the  summarized gene expressions.

    Parameters
    ----------
    num_genes
        Number of genes to be selected.
    top
        If set to :obj:`True`, then use the genes with highest values of the specified gene summary stats.
    mode
        Summarization mode. Available options are ``[sum|var|cv|rv]``. ``sum`` calculates the sum of expression values,
        ``var`` calculates the variance of the expression values, ``cv`` uses the coefficient of variation (std / mean
        ), and ``rv`` uses the relative variance (var / mean).
    channel
        Which channel, more specificailly, ``layers``, to use. Use the default ``.X`` if not set. If ``channel`` is
        specified, then need to specify ``channel_type`` to be ``layers`` as well.
    channel_type
        Type of channels specified. Only allow ``None`` (the default setting) or ``layers`` (when ``channel`` is
        specified).
    whitelist_indicators
        A list of (or a single) :obj:`.var` columns that indicates the genes to be excluded from the filtering process.
        Note that these genes will still be used in the summary stats computation, and thus will still contribute to the
        threshold percentile. If not set, then no genes will be excluded from the filtering process.
    add_n_counts
        Whether to add n_counts
    add_n_cells
        Whether to add n_cells

    """

    _DISPLAY_ATTRS = ("num_genes", "top", "mode")

    def __init__(
        self,
        num_genes: int = 1000,
        top: bool = True,
        *,
        mode: GeneSummaryMode = "cv",
        channel: Optional[str] = None,
        channel_type: Optional[str] = "X",
        whitelist_indicators: Optional[Union[str, List[str]]] = None,
        add_n_counts=False,
        add_n_cells=False,
        **kwargs,
    ):
        super().__init__(
            mode=mode,
            channel=channel,
            channel_type=channel_type,
            whitelist_indicators=whitelist_indicators,
            add_n_counts=add_n_counts,
            add_n_cells=add_n_cells,
            **kwargs,
        )
        self.num_genes = num_genes
        self.top = top

    def _get_preserve_mask(self, gene_summary):
        total_num_genes = gene_summary.size
        if self.num_genes >= total_num_genes:
            # raise ValueError(f"{self.num_genes=!r} > total number of genes: {total_num_genes}")
            self.logger.warning(f"{self.num_genes=!r} > total number of genes: {total_num_genes}")
            self.num_genes = total_num_genes
        sorted_idx = gene_summary.argsort()
        selected_idx = sorted_idx[-self.num_genes:] if self.top else sorted_idx[:self.num_genes]
        mask = np.zeros(total_num_genes, dtype=bool)
        mask[selected_idx] = True
        return mask


@register_preprocessor("filter", "gene")
class FilterGenesMarker(BaseTransform):
    """Select marker genes based on log fold-change.

    Parameters
    ----------
    ct_profile_channel
        Name of the ``.varm`` channel that contains the cell-topic profile which will be used to compute the log
        fold-changes for each cell-topic (e.g., cell type).
    subset
        If set to :obj:`True`, then inplace subset the variables to only contain the markers.
    label
        If set, e.g., to :obj:`'marker'`, then save the marker indicator to the :obj:`.obs` column named as
        :obj:`marker`.
    threshold
        Threshold value of the log fol-change above which the gene will be considered as a marker.
    eps
        A small value that prevents taking log of zeros.

    """

    _DISPLAY_ATTRS = ("ct_profile_channel", "subset", "threshold", "eps")

    def __init__(
        self,
        *,
        ct_profile_channel: str = "CellTopicProfile",
        subset: bool = True,
        label: Optional[str] = None,
        threshold: float = 1.25,
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ct_profile_channel = ct_profile_channel
        self.subset = subset
        self.label = label
        self.threshold = threshold
        self.eps = eps

    @staticmethod
    def get_marker_genes(
        ct_profile: np.ndarray,  # gene x celltype
        cell_types: List[str],
        genes: List[str],
        *,
        threshold: float = 1.25,
        eps: float = 1e-6,
        logger: Logger = default_logger,
    ) -> Tuple[List[str], pd.DataFrame]:
        if (num_cts := len(cell_types)) < 2:
            raise ValueError(f"Need at least two cell types to find marker genes, got {num_cts}:\n{cell_types}")

        # Find marker genes for each cell type
        marker_gene_ind_df = pd.DataFrame(False, index=genes, columns=cell_types)
        for i, ct in enumerate(cell_types):
            others = [j for j in range(num_cts) if j != i]
            log_fc = np.log(ct_profile[:, i] + eps) - np.log(ct_profile[:, others].mean(1) + eps)
            markers_idx = np.where(log_fc > threshold)[0]

            if markers_idx.size > 0:
                marker_gene_ind_df.iloc[markers_idx, i] = True
                markers = marker_gene_ind_df.iloc[markers_idx].index.tolist()
                logger.info(f"Found {len(markers):,} marker genes for cell type {ct!r}")
                logger.debug(f"{markers=}")
            else:
                logger.info(f"No marker genes found for cell type {ct!r}")

        # Combine all marker genes
        is_marker = marker_gene_ind_df.max(1)
        marker_genes = is_marker[is_marker].index.tolist()
        logger.info(f"Total number of marker genes found: {len(marker_genes):,}")
        logger.debug(f"{marker_genes=}")

        return marker_genes, marker_gene_ind_df

    def __call__(self, data):
        ct_profile_df = data.get_feature(channel=self.ct_profile_channel, channel_type="varm", return_type="default")
        ct_profile = ct_profile_df.values
        cell_types = ct_profile_df.columns.tolist()
        genes = ct_profile_df.index.tolist()
        marker_genes, marker_gene_ind_df = self.get_marker_genes(ct_profile, cell_types, genes, eps=self.eps,
                                                                 threshold=self.threshold, logger=self.logger)

        # Save marker gene info to data
        data.data.varm[self.out] = marker_gene_ind_df
        if self.label is not None:
            data.data.var[self.label] = marker_gene_ind_df.max(1)

        if self.subset:  # inplace subset the variables
            data.data._inplace_subset_var(marker_genes)


@register_preprocessor("filter", "gene")
class FilterGenesRegression(BaseTransform):
    """Select genes based on regression.

    Parameters
    ----------
    method
        What regression based gene selection methtod to use. Supported options are: ``"enclasc"``, ``"seurat3"``, and
        ``"scmap"``.
    num_genes
        Number of genes to select.

    Note
    ----
    The implementation is adapted from the EnClaSC GitHub repo: https://github.com/xy-chen16/EnClaSC

    Reference
    ---------
    https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03679-z

    """
    _DISPLAY_ATTRS = ("num_genes", )

    def __init__(self, method: str = "enclasc", num_genes: int = 1000, *, channel: Optional[str] = None,
                 mod: Optional[str] = None, skip_count_check: bool = False, **kwargs):
        super().__init__(**kwargs)

        self.num_genes = num_genes
        self.channel = channel
        self.mod = mod
        self.method = method
        self.skip_count_check = skip_count_check

    def __call__(self, data):
        feat = data.get_feature(return_type="numpy", channel=self.channel, mod=self.mod)

        if not self.skip_count_check and np.mod(feat, 1).sum():
            warnings.warn("Expecting count data as input, but the input feature matrix does not appear to be count."
                          "Please make sure the input is indeed a count matrix.")

        func_dict = {"enclasc": self._filter_enclasc, "seurat3": self._filter_seurat3, "scmap": self._filter_scmap}
        if (filter_func := func_dict.get(self.method)) is None:
            raise ValueError(f"Unknown method {self.method}, supported options are: {list(func_dict)}.")
        if self.num_genes >= feat.shape[1]:
            self.logger.warning(f"{self.num_genes=!r} > total number of genes: {feat.shape[1]}")
            self.num_genes = feat.shape[1]
        # data.data.obsm[self.out] = filter_func(feat, self.num_genes)
        gene_names = data.data.var_names[filter_func(feat, self.num_genes)]
        data.data._inplace_subset_var(gene_names)
        return data

    def _filter_enclasc(self, feat: np.ndarray, num_genes: int = 2000, logger: Logger = default_logger,
                        no_check: bool = False) -> np.ndarray:
        logger.info("Start generating cell features using EnClaSC")
        num_feat = feat.shape[1]
        scores = np.zeros(num_feat) - 100

        feat_mean = feat.mean(0)
        drop_feat = (feat == 0).mean(0)
        select_index = (0 < drop_feat) & (drop_feat < 1)

        x1 = feat_mean[select_index].reshape(-1, 1)
        x2 = drop_feat[select_index].reshape(-1, 1)
        y = np.log(feat_mean + 1)[select_index].reshape(-1, 1)

        y_pred = LinearRegression(n_jobs=8).fit(x2, y).predict(x2)
        scores[select_index] = (2 * y - y_pred - x1).ravel()
        feat_index = np.argpartition(scores, -num_genes)[-num_genes:]
        return feat_index

    def _filter_seurat3(self, feat: np.ndarray, num_genes: int = 2000, logger: Logger = default_logger,
                        no_check: bool = False) -> np.ndarray:
        logger.info("Start generating cell features using Seurat v3.0")

        feat_mean_log = np.log(feat.mean(0) + 1)
        feat_var_log = np.log(feat.var(0) + 1)
        x = PolynomialFeatures(degree=2).fit_transform(feat_mean_log.reshape(-1, 1))

        y_pred = LinearRegression().fit(x, feat_var_log).predict(x)
        scores = (feat_var_log - y_pred).ravel()
        feat_index = np.argpartition(scores, -num_genes)[-num_genes:]
        return feat_index

    def _filter_scmap(self, feat: np.ndarray, num_genes: int = 2000, logger: Logger = default_logger,
                      no_check: bool = False) -> np.ndarray:
        logger.info("Start generating cell features using scmap")

        num_feat = feat.shape[1]
        scores = np.zeros(num_feat) - 100

        feat_mean = feat.mean(0)
        drop_feat = (feat == 0).mean(0)
        select_index = (0 < drop_feat) & (drop_feat < 1)

        x = np.log(feat_mean[select_index] + 1).reshape(-1, 1) * np.log(2.7) / np.log(2)
        y = np.log(drop_feat[select_index] * 100).reshape(-1, 1) * np.log(2.7) / np.log(2)

        y_pred = LinearRegression().fit(x, y).predict(x)
        scores[select_index] = (y - y_pred).ravel()
        feat_index = np.argpartition(scores, -num_genes)[-num_genes:]
        return feat_index


@register_preprocessor("filter", "gene")
class FilterGenesMarkerGini(BaseTransform):
    """Select marker genes based on Gini coefficient.

    Identfy marker genes for all clusters in a one vs all manner based on Gini coefficients, a measure for inequality.

    Parameters
    ----------
    ct_profile_channel
        Name of the ``.varm`` channel that contains the cell-topic profile which will be used to compute the log
        fold-changes for each cell-topic (e.g., cell type).
    ct_profile_detection_channel
        Name of the ``.varm`` channel that contains the cell-topic profile nums which greater than some value which
        will be used to compute the log fold-changes for each cell-topic (e.g., cell type).
    subset
        If set to :obj:`True`, then inplace subset the variables to only contain the markers.
    label
        If set, e.g., to :obj:`'marker'`, then save the marker indicator to the :obj:`.obs` column named as
        :obj:`marker`.

    Reference
    ---------
    https://genomebiology.biomedcentral.com/articles/10.1186/s13059-016-1010-4?ref=https://githubhelp.com

    """

    def __init__(
        self,
        *,
        ct_profile_channel: str = "CellGiottoTopicProfile",
        ct_profile_detection_channel: str = "CellGiottoDetectionTopicProfile",
        subset: bool = True,
        label: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ct_profile_channel = ct_profile_channel
        self.ct_profile_detection_channel = ct_profile_detection_channel
        self.subset = subset
        self.label = label

    def __call__(
        self,
        data,
        logger: Logger = default_logger,
    ):
        ct_profile_df = data.get_feature(channel=self.ct_profile_channel, channel_type="varm", return_type="default")
        ct_profile_detection_df = data.get_feature(channel=self.ct_profile_detection_channel, channel_type="varm",
                                                   return_type="default")
        cell_type_nums_df = data.get_feature(channel="CellTypeNums", channel_type="uns", return_type="default")

        ct_profile = ct_profile_df.values
        ct_profile_detection = ct_profile_detection_df.values
        cell_types = ct_profile_df.columns.tolist()
        genes = ct_profile_df.index.tolist()
        marker_gene_ind_df = pd.DataFrame(False, index=genes, columns=cell_types)

        ans_gene = []
        if (num_cts := len(cell_types)) < 2:
            raise ValueError(f"Need at least two cell types to find marker genes, got {num_cts}:\n{cell_types}")

        for i, ct in enumerate(cell_types):
            other_ct_profile = np.zeros_like(ct_profile[:, i])
            other_detection_ct_profile = np.zeros_like(ct_profile[:, i])
            other_sum = 0

            for j in range(num_cts):
                if j != i:
                    other = cell_type_nums_df.loc[cell_types[j], "nums"]
                    other_ct_profile += ct_profile[:, j] * other
                    other_detection_ct_profile += ct_profile_detection[:, j] * other
                    other_sum += other
            other_ct_profile = other_ct_profile / other_sum
            other_detection_ct_profile = other_detection_ct_profile / other_sum
            top_genes_scores_filtered = get_marker_genes_giotto(ct_profile[:, i], other_ct_profile,
                                                                ct_profile_detection[:, i], other_detection_ct_profile,
                                                                genes=genes)
            markers_idx = np.array(top_genes_scores_filtered.index)
            top_genes_scores_filtered["cellType"] = ct
            ans_gene.append(top_genes_scores_filtered)
            if markers_idx.size > 0:
                marker_gene_ind_df.iloc[markers_idx, i] = True
                markers = marker_gene_ind_df.iloc[markers_idx].index.tolist()
                logger.info(f"Found {len(markers):,} marker genes for cell type {ct!r}")
                logger.debug(f"{markers=}")
            else:
                logger.info(f"No marker genes found for cell type {ct!r}")

        # Combine all marker genes
        is_marker = marker_gene_ind_df.max(1)
        marker_genes = is_marker[is_marker].index.tolist()

        # Save marker gene info to data
        data.data.uns[self.out] = pd.concat(ans_gene, axis=0)
        if self.label is not None:
            data.data.var[self.label] = pd.concat(marker_gene_ind_df.max(1))

        if self.subset:  # inplace subset the variables
            data.data._inplace_subset_var(marker_genes)


def get_marker_genes_giotto(group1, group2, group_detection_1, group_detection_2, min_expr_gini_score=0.2,
                            min_det_gini_score=0.2, rank_score=1, min_genes=5, genes=None):
    gene_nums = group1.shape[0]
    gene_detection_gini_score = np.zeros((2, gene_nums))
    gene_gini_score = np.zeros((2, gene_nums))
    gene_rank_score = np.zeros((2, gene_nums))
    expressions = np.zeros((2, gene_nums))
    detections = np.zeros((2, gene_nums))
    gene_detection_rank_score = np.zeros((2, gene_nums))
    scaler = MinMaxScaler(feature_range=(0.1, 1))  # inverse
    for i in range(gene_nums):
        gene_gini_score[:, i] = [gini_func([group1[i], group2[i]])] * 2
        expressions[:, i] = [group1[i], group2[i]]
        gene_detection_gini_score[:, i] = [gini_func([group_detection_1[i], group_detection_2[i]])] * 2
        detections[:, i] = [group_detection_1[i], group_detection_2[i]]

        gene_rank_score[:, i] = rankdata(np.array([group1[i], group2[i]]))  # inverse
        gene_detection_rank_score[:, i] = rankdata(np.array([group_detection_1[i], group_detection_2[i]]))
    gene_rank_score = scaler.fit_transform(gene_rank_score)
    gene_detection_rank_score = scaler.fit_transform(gene_detection_rank_score)
    ans_score = (gene_detection_gini_score * gene_gini_score * gene_rank_score * gene_detection_rank_score)[0]
    ans_rank = np.argsort(np.argsort(-ans_score)) + 1
    ans_df = pd.DataFrame(
        False, index=[i for i in range(gene_nums)], columns=[
            "ans_score", "ans_rank", "expression", "detection", "expression_gini", "detection_gini", "gene_rank_score",
            "gene_detection_rank_score", "gene_name"
        ])
    ans_df.loc[:, ["ans_score"]] = ans_score
    ans_df.loc[:, ["ans_rank"]] = ans_rank
    ans_df.loc[:, ["expression"]] = expressions[0]
    ans_df.loc[:, ["detection"]] = detections[0]
    ans_df.loc[:, ["expression_gini"]] = gene_gini_score[0]
    ans_df.loc[:, ["detection_gini"]] = gene_detection_gini_score[0]
    ans_df.loc[:, ["gene_rank_score"]] = gene_rank_score[0]
    ans_df.loc[:, ["gene_detection_rank_score"]] = gene_detection_rank_score[0]
    ans_df.loc[:, ["gene_name"]] = genes
    # Filter on combined rank or individual ranks
    top_genes_scores = ans_df[(ans_df['ans_rank'] <= min_genes) | (ans_df['gene_rank_score'] <= rank_score) &
                              (ans_df['gene_detection_rank_score'] <= rank_score)]

    # Further filter on expression and detection gini score
    top_genes_scores_filtered = top_genes_scores[(top_genes_scores['ans_rank'] <= min_genes) |
                                                 (top_genes_scores['expression'] > min_expr_gini_score) &
                                                 (top_genes_scores['detection'] > min_det_gini_score)]
    return top_genes_scores_filtered


def gini_func(x, weights=None):
    if weights is None:
        weights = np.ones(len(x))
    dataset = np.column_stack((x, weights))
    ord_x = np.argsort(x)
    dataset_ord = dataset[ord_x]
    x = dataset_ord[:, 0]
    weights = dataset_ord[:, 1]

    N = np.sum(weights)
    xw = x * weights
    C_i = np.cumsum(weights)
    num_1 = np.sum(xw * C_i)
    num_2 = np.sum(xw)
    num_3 = np.sum(xw * weights)

    G_num = (2 / N**2) * num_1 - (1 / N) * num_2 - (1 / N**2) * num_3

    t_neg = xw[xw <= 0]
    T_neg = np.sum(t_neg)
    T_pos = np.sum(xw) + np.abs(T_neg)

    n_RSV = 2 * (T_pos + np.abs(T_neg)) / N
    mean_RSV = n_RSV / 2

    G_RSV = G_num / mean_RSV

    return G_RSV


@register_preprocessor("filter", "gene")
class FilterGenesScanpyOrder(BaseTransform):
    """Scanpy filtering gene transformation with additional options.

    Parameters
    ----------
    order
        Order of (min_counts, min_cells, max_counts, max_cells). For example,
        ``["min_counts", "min_cells", "max_counts", "max_cells"]`` or ``["max_counts", "min_cells"]``.
        If not set, will be set by default to ``["min_counts", "min_cells", "max_counts", "max_cells"]``.
    min_counts
        Minimum number of counts required for a gene to be kept.
    min_cells
        Minimum number (or ratio) of cells required for a gene to be kept.
    max_counts
        Maximum number of counts required for a gene to be kept.
    max_cells
        Maximum number (or ratio) of cells required for a gene to be kept.
    split_name
        Which split to be used for filtering.
    channel
        Channel to be used for filtering.
    channel_type
        Channel type to be used for filtering.
    add_n_counts
        Whether to add n_counts
    add_n_cells
        Whether to add n_cells

    """

    def __init__(
        self,
        order: Optional[List[str]] = None,
        min_counts: Optional[int] = None,
        min_cells: Optional[Union[float, int]] = None,
        max_counts: Optional[int] = None,
        max_cells: Optional[Union[float, int]] = None,
        split_name: Optional[str] = None,
        channel: Optional[str] = None,
        channel_type: Optional[str] = "X",
        add_n_counts=True,
        add_n_cells=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filter_genes_order = default(
            order,
            ["min_counts", "min_cells", "max_counts", "max_cells"],
        )
        self.logger.info(f"Filter genes order: {self.filter_genes_order}")
        geneParameterDict = {
            "min_counts": min_counts,
            "min_cells": min_cells,
            "max_counts": max_counts,
            "max_cells": max_cells
        }
        self.add_n_counts = add_n_counts
        self.add_n_cells = add_n_cells
        if not set(self.filter_genes_order).issubset(set(geneParameterDict.keys())):
            raise KeyError(f"An order should be in {geneParameterDict.keys()}")
        self.geneScanpyOrderDict = {}
        for key in geneParameterDict.keys():
            if key in self.filter_genes_order:
                if key in self.filter_genes_order:
                    if key in ("min_counts", "max_counts") and self.add_n_counts:
                        key_n_counts = "n_counts"
                    elif key in ("min_cells", "max_cells") and self.add_n_cells:
                        key_n_counts = "n_cells"
                    else:
                        key_n_counts = None
                self.geneScanpyOrderDict[key] = FilterGenesScanpy(
                    **{key: geneParameterDict[key]},
                    split_name=split_name,
                    channel=channel,
                    channel_type=channel_type,
                    key_n_counts=key_n_counts,
                    **kwargs,
                )
            else:
                self.logger.warning(f"{key} not in order,It makes no sense to set {key}")

    def __call__(self, data: Data):
        for parameter in self.filter_genes_order:
            geneScanpyOrder = self.geneScanpyOrderDict[parameter]
            geneScanpyOrder(data)


@register_preprocessor("filter", "gene")
class HighlyVariableGenesRawCount(AnnDataTransform):
    """Filter for highly variable genes using raw count matrix.

    Parameters
    ----------
    layer
        If provided, then use `data.data.layers[layer]` for expression values instead of the
        default ``data.data.X``.
    n_top_genes
        Number of highly-variable genes to keep.
    span
        The fraction of the data (cells) used when estimating the variance in the loess
        model fit if `flavor="seurat_v3"`.
    subset
        Inplace subset to highly-variable genes if `True` otherwise merely indicate
        highly variable genes.
    inplace
        Whether to place calculated metrics in `.var` or return them.
    batch_key
        If specified, highly-variable genes are selected within each batch separately and merged.
        This simple process avoids the selection of batch-specific genes and acts as a
        lightweight batch correction method. For all flavors, genes are first sorted
        by how many batches they are a HVG. For dispersion-based flavors ties are broken
        by normalized dispersion. If `flavor = "seurat_v3"`, ties are broken by the median
        (across batches) rank based on within-batch normalized variance.
    check_values
        Check if counts in selected layer are integers. A Warning is returned if set to True.
        Only used if `flavor="seurat_v3"`.

    See also
    --------
    This is a wrapper for
    https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.highly_variable_genes.html

    """

    def __init__(self, layer: Optional[str] = None, n_top_genes: Optional[int] = 1000, span: Optional[float] = 0.3,
                 subset: bool = True, inplace: bool = True, batch_key: Optional[str] = None, check_values: bool = True,
                 **kwargs):
        super().__init__(sc.pp.highly_variable_genes, layer=layer, n_top_genes=n_top_genes, batch_key=batch_key,
                         check_values=check_values, span=span, subset=subset, inplace=inplace, flavor="seurat_v3",
                         **kwargs)

    def __call__(self, data):
        adata = data.data
        if adata.X.shape[1] == 0:
            raise ValueError("Gene dimension is 0")  #Prevent kernel crash
        self.logger.info("Expects count data")
        return super().__call__(data)


@register_preprocessor("filter", "gene")
class HighlyVariableGenesLogarithmizedByTopGenes(AnnDataTransform):
    """Filter for highly variable genes based on top genes.

    Parameters
    ----------
    layer
        If provided, then use data.data.layers[layer]` for expression values instead of the
        default `data.data.X`.
    n_top_genes
        Number of highly-variable genes to keep.
    n_bins
        Number of bins for binning the mean gene expression. Normalization is
        done with respect to each bin. If just a single gene falls into a bin,
        the normalized dispersion is artificially set to 1. You'll be informed
        about this if you set `settings.verbosity = 4`.
    flavor
        Choose the flavor for identifying highly variable genes. For the dispersion
        based methods in their default workflows, Seurat passes the cutoffs whereas
        Cell Ranger passes `n_top_genes`.
    subset
        Inplace subset to highly-variable genes if `True` otherwise merely indicate
        highly variable genes.
    inplace
        Whether to place calculated metrics in `.var` or return them.
    batch_key
        If specified, highly-variable genes are selected within each batch separately and merged.
        This simple process avoids the selection of batch-specific genes and acts as a
        lightweight batch correction method. For all flavors, genes are first sorted
        by how many batches they are a HVG. For dispersion-based flavors ties are broken
        by normalized dispersion. If `flavor = "seurat_v3"`, ties are broken by the median
        (across batches) rank based on within-batch normalized variance.

    See also
    --------
    This is a wrapper for
    https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.highly_variable_genes.html

    """

    def __init__(self, layer: Optional[str] = None, n_top_genes: Optional[int] = 1000, n_bins: int = 20,
                 flavor: Literal["seurat", "cell_ranger"] = "seurat", subset: bool = True, inplace: bool = True,
                 batch_key: Optional[str] = None, **kwargs):
        super().__init__(sc.pp.highly_variable_genes, layer=layer, n_top_genes=n_top_genes, n_bins=n_bins,
                         flavor=flavor, subset=subset, inplace=inplace, batch_key=batch_key, **kwargs)
        self.logger.info("Expects logarithmized data")


@register_preprocessor("filter", "gene")
@deprecated(msg="will be replaced by builtin bypass mechanism in pipeline")
class FilterGenesPlaceHolder(BaseTransform):
    """Used as a placeholder to skip the process."""

    def __init__(self, split_name: Optional[str] = None, channel: Optional[str] = None,
                 channel_type: Optional[str] = "X", add_n_counts=True, add_n_cells=True, **kwargs):
        super().__init__(**kwargs)
        self.split_name = split_name
        self.channel = channel
        self.channel_type = channel_type
        self.add_n_counts = add_n_counts
        self.add_n_cells = add_n_cells

    def __call__(self, data: Data) -> Data:
        x = data.get_feature(return_type="numpy", split_name=self.split_name, channel=self.channel,
                             channel_type=self.channel_type)
        n_counts = np.sum(x, axis=0)
        n_cells = np.sum(x > 0, axis=0)
        if self.add_n_counts:
            self.logger.warning(f"n_counts will be added to the var of data")
            data.data.var["n_counts"] = n_counts
        if self.add_n_cells:
            self.logger.warning(f"n_cells will be added to the var of data")
            data.data.var["n_cells"] = n_cells
        return data


@register_preprocessor("filter", "gene")
@deprecated(msg="will be replaced by builtin bypass mechanism in pipeline")
class FilterGenesNumberPlaceHolder(BaseTransform):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, data: Data) -> Data:
        return data


@register_preprocessor("filter", "gene")
class HighlyVariableGenesLogarithmizedByMeanAndDisp(AnnDataTransform):
    """Filter for highly variable genes based on mean and dispersion.

    Parameters
    ----------
    layer
        If provided, then use data.data.layers[layer]` for expression values instead of the
        default `data.data.X`.
    min_mean
        min_mean
    max_mean
        max_mean
    min_disp
        min_disp
    max_disp
        max_disp
    n_bins
        Number of bins for binning the mean gene expression. Normalization is
        done with respect to each bin. If just a single gene falls into a bin,
        the normalized dispersion is artificially set to 1. You'll be informed
        about this if you set `settings.verbosity = 4`.
    flavor
        Choose the flavor for identifying highly variable genes. For the dispersion
        based methods in their default workflows, Seurat passes the cutoffs whereas
        Cell Ranger passes `n_top_genes`.
    subset
        Inplace subset to highly-variable genes if `True` otherwise merely indicate
        highly variable genes.
    inplace
        Whether to place calculated metrics in `.var` or return them.
    batch_key
        If specified, highly-variable genes are selected within each batch separately and merged.
        This simple process avoids the selection of batch-specific genes and acts as a
        lightweight batch correction method. For all flavors, genes are first sorted
        by how many batches they are a HVG. For dispersion-based flavors ties are broken
        by normalized dispersion. If `flavor = "seurat_v3"`, ties are broken by the median
        (across batches) rank based on within-batch normalized variance.

    See also
    --------
    This is a wrapper for
    https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.highly_variable_genes.html

    """

    def __init__(self, layer: Optional[str] = None, min_disp: Optional[float] = 0.5, max_disp: Optional[float] = np.inf,
                 min_mean: Optional[float] = 0.0125, max_mean: Optional[float] = 3, n_bins: int = 20,
                 flavor: Literal["seurat", "cell_ranger"] = "seurat", subset: bool = True, inplace: bool = True,
                 batch_key: Optional[str] = None, **kwargs):
        super().__init__(sc.pp.highly_variable_genes, layer=layer, min_disp=min_disp, max_disp=max_disp,
                         min_mean=min_mean, max_mean=max_mean, n_bins=n_bins, flavor=flavor, subset=subset,
                         inplace=inplace, batch_key=batch_key, **kwargs)
        self.logger.info("Expects logarithmized data")


@register_preprocessor("filter", "cell")
@deprecated(msg="will be replaced by builtin bypass mechanism in pipeline")
class FilterCellsPlaceHolder(BaseTransform):
    """Used as a placeholder to skip the process."""

    def __init__(self, split_name: Optional[str] = None, channel: Optional[str] = None,
                 channel_type: Optional[str] = "X", add_n_counts=True, add_n_genes=True, **kwargs):
        super().__init__(**kwargs)
        self.split_name = split_name
        self.channel = channel
        self.channel_type = channel_type
        self.add_n_counts = add_n_counts
        self.add_n_genes = add_n_genes

    def __call__(self, data: Data) -> Data:
        x = data.get_feature(return_type="numpy", split_name=self.split_name, channel=self.channel,
                             channel_type=self.channel_type)
        n_counts = np.sum(x, axis=1)
        n_genes = np.sum(x > 0, axis=1)
        if self.add_n_counts:
            self.logger.warning(f"n_counts will be added to the obs of data")
            data.data.obs["n_counts"] = n_counts
        if self.add_n_genes:
            self.logger.warning(f"n_genes will be added to the obs of data")
            data.data.obs["n_genes"] = n_genes

        return data


@register_preprocessor("filter", "cell")
class FilterCellsScanpyOrder(BaseTransform):
    """Scanpy filtering cell transformation with additional options.

    Allow passing gene counts as ratio

    Parameters
    ----------
    order
        Order of (min_counts, min_cells, max_counts, max_cells). For example,
        ``["min_counts", "min_genes", "max_counts", "max_genes"]`` or ``["max_counts", "min_genes"]``.
        If not set, will be set by default to ``["min_counts", "min_genes", "max_counts", "max_genes"]``.
    min_counts
        Minimum number of counts required for a cell to be kept.
    min_genes
        Minimum number (or ratio) of genes required for a cell to be kept.
    max_counts
        Maximum number of counts required for a cell to be kept.
    max_genes
        Maximum number (or ratio) of genes required for a cell to be kept.
    split_name
        Which split to be used for filtering.
    channel
        Channel to be used for filtering.
    channel_type
        Channel type to be used for filtering.
    add_n_counts
        Whether to add n_counts
    add_n_genes
        Whether to add n_genes

    """

    def __init__(self, order: Optional[List[str]] = None, min_counts: Optional[int] = None,
                 min_genes: Optional[Union[float, int]] = None, max_counts: Optional[int] = None,
                 max_genes: Optional[Union[float, int]] = None, split_name: Optional[str] = None,
                 channel: Optional[str] = None, channel_type: Optional[str] = "X", add_n_counts=True, add_n_genes=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.filter_cells_order = default(order, ["min_counts", "min_genes", "max_counts", "max_genes"])
        self.logger.info(f"Filter cells order: {self.filter_cells_order}")
        cellParameterDict = {
            "min_counts": min_counts,
            "min_genes": min_genes,
            "max_counts": max_counts,
            "max_genes": max_genes
        }
        self.add_n_counts = add_n_counts
        self.add_n_genes = add_n_genes
        if not set(self.filter_cells_order).issubset(set(cellParameterDict.keys())):
            raise KeyError(f"An order should be in {cellParameterDict.keys()}")
        self.cellScanpyOrderDict = {}
        for key in cellParameterDict.keys():
            if key in self.filter_cells_order:
                if key in ("min_counts", "max_counts") and self.add_n_counts:
                    key_n_counts = "n_counts"
                elif key in ("min_genes", "max_genes") and self.add_n_genes:
                    key_n_counts = "n_genes"
                else:
                    key_n_counts = None
                self.cellScanpyOrderDict[key] = FilterCellsScanpy(**{key:
                                                                     cellParameterDict[key]}, split_name=split_name,
                                                                  channel=channel, channel_type=channel_type,
                                                                  key_n_counts=key_n_counts, **kwargs)
            else:
                self.logger.warning(f"{key} not in order,It makes no sense to set {key}")

    def __call__(self, data: Data):
        for parameter in self.filter_cells_order:
            cellScanpyOrder = self.cellScanpyOrderDict[parameter]
            cellScanpyOrder(data)
