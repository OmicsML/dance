import numpy as np

from dance.exceptions import DevError
from dance.transforms.base import BaseTransform
from dance.typing import Literal, Optional


class FilterGenesPercentile(BaseTransform):
    """Filter genes based on percentiles of the summarized gene expressions."""

    _DISPLAY_ATTRS = ("min_val", "max_val", "mode")
    _MODES = ["sum", "cv"]

    def __init__(self, min_val: Optional[float] = 1, max_val: Optional[float] = 99, mode: Literal["sum", "cv"] = "sum",
                 *, channel: Optional[str] = None, channel_type: Optional[str] = None, **kwargs):
        """Initialize FilterGenesPercentile.

        Parameters
        ----------
        min_val
            Minimum percentile of the summarized expression value below which the genes will be discarded.
        max_val
            Maximum percentile of the summarized expression value above which the genes will be discarded.
        mode
            Summarization mode. Available options are ``[sum|cv]``. ``sum`` calculates the sum of expression values,
            ``cv`` uses the coefficient of variation (std / mean).
        channel
            Which channel, more specificailly, ``layers``, to use. Use the default ``.X`` if not set. If ``channel`` is
            specified, then need to specify ``channel_type`` to be ``layers`` as well.
        channel_type
            Type of channels specified. Only allow ``None`` (the default setting) or ``layers`` (when ``channel`` is
            specified).

        """
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
