import numpy as np

from dance.transforms.base import BaseTransform
from dance.typing import Optional


class FilterGenesPercentile(BaseTransform):
    """Filter genes based on percentiles of the total expression of genes."""

    _DISPLAY_ATTRS = ("min_val", "max_val")

    def __init__(self, min_val: Optional[float] = 1, max_val: Optional[float] = 99, channel: Optional[str] = None,
                 channel_type: Optional[str] = None, **kwargs):
        """Initialize FilterGenesPercentile.

        Parameters
        ----------
        min_val
            Minimum percentile of the total expression value below which the genes will be discarded.
        max_val
            Maximum percentile of the total expression value above which the genes will be discarded.
        channel
            Which channel, more specificailly, `layers`, to use. Use the default `.X` if not set. If `channel` is
            specified, then need to specify `channel_type` to be `layers` as well.
        channel_type
            Type of channels specified. Only allow `None` (the default setting) or `layers` (when `channel` is
            specified).

        """
        super().__init__(**kwargs)

        if (channel is not None) and (channel_type != "layers"):
            raise ValueError(f"Only X layers is available for filtering genes, specified {channel_type=!r}")

        self.min_val = min_val
        self.max_val = max_val
        self.channel = channel
        self.channel_type = channel_type

    def __call__(self, data):
        x = data.get_feature(return_type="default", channel=self.channel, channel_type=self.channel_type)
        gene_sum = np.array(x.sum(0)).ravel()

        percentile_lo = np.percentile(gene_sum, self.min_val)
        percentile_hi = np.percentile(gene_sum, self.max_val)
        mask = np.logical_and(gene_sum >= percentile_lo, gene_sum <= percentile_hi)
        self.logger.info(f"Filtering genes based on total expression percentiles in layer {self.channel!r}")
        self.logger.info(f"{mask.size - mask.sum()} genes removed ({percentile_lo=:.2e}, {percentile_hi=:.2e})")

        data._data = data.data[:, mask].copy()
