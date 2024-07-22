from pprint import pformat
from typing import Optional

import mudata as md

from dance import logger
from dance.data.base import Data
from dance.registry import register_preprocessor
from dance.transforms.base import BaseTransform
from dance.typing import Any, Dict, Tuple
from dance.utils import hexdigest


@register_preprocessor("misc")
class Compose(BaseTransform):
    """Compose transformation by combining several transfomration objects.

    Parameters
    ----------
    transforms
        Transformation objects.
    use_master_log_level
        If set to ``True``, then reset all transforms' loggers to use :then reset all transforms' loggers to use
        ``log_level`` option passed to this :class:`Compose` object.

    Notes
    -----
    The order in which the ``transform`` object are passed will be exactly the order in which they will be applied to
    the data object.

    """

    def __init__(self, *transforms: Tuple[BaseTransform, ...], use_master_log_level: bool = True, **kwargs):
        super().__init__(**kwargs)

        # Check type
        failed_list = []
        for transform in transforms:
            if not isinstance(transform, BaseTransform):
                failed_list.append(transform)
        if failed_list:
            failed_list_str = "\n".join(["\t{i!r}: {type(i)!r}" for i in failed_list])
            raise TypeError("Expect all transform objects to be inherited from BaseTransform. The following "
                            f"(n={len(failed_list)}) have incorrect types:\n{failed_list_str}")

        self.transforms = transforms

        # Set log level using master log level from the Compose object
        if use_master_log_level:
            for transform in transforms:
                transform.log_level = self.log_level
                transform.logger.setLevel(self.log_level)

    def __repr__(self):
        transform_repr_str = ",\n  ".join(map(repr, self.transforms))
        return f"Compose(\n  {transform_repr_str},\n)"

    def __getitem__(self, idx: int, /):
        return self.transforms[idx]

    def hexdigest(self) -> str:
        hexdigests = [i.hexdigest() for i in self.transforms]
        md5_hash = hexdigest("".join(hexdigests))
        logger.debug(f"{hexdigest=}, {md5_hash=}")
        return md5_hash

    def __call__(self, data):
        self.logger.info(f"Applying composed transformations:\n{self!r}")
        for transform in self.transforms:
            transform(data)


@register_preprocessor("misc")
class SetConfig(BaseTransform):
    """Set configuration options of a dance data object.

    Parameters
    ----------
    config_dict
        Dance data object configuration dictionary. See :meth:`~dance.data.base.BaseData.set_config_from_dict`.

    """

    _DISPLAY_ATTRS = ("config_dict", )

    def __init__(self, config_dict: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.config_dict = config_dict

    def __call__(self, data):
        self.logger.info(f"Updating the dance data object config options:\n{pformat(self.config_dict)}")
        data.set_config_from_dict(self.config_dict)


@register_preprocessor("misc")
class SaveRaw(BaseTransform):
    """Save raw data.

    See :meth:`anndata.AnnData.raw` for more information.

    Parameters
    ----------
    exist_ok
        If set to False, then raise an exception if the :obj:`raw` attribute is already set.

    """

    def __init__(self, exist_ok: bool = False, **kwargs):
        super().__init__(**kwargs)

        self.exist_ok = exist_ok

    def __call__(self, data):
        self.logger.info("Saving data to ``.raw``")
        if data.data.raw is not None:
            if self.exist_ok:
                self.logger.warning("Overwriting raw content...")
            else:
                raise AttributeError(f"Raw data attribute already exist and cannot be overwritten.\n{data}"
                                     f"If you wish to overwrite, set 'exist_ok' to True.")
        data.data.raw = data.data.copy()


@register_preprocessor("misc")
class UpdateRaw(BaseTransform):
    """Update raw data.

    Some data may select genes again after normalizing.
    At this time, the original raw_data needs to be modified to match the filtered data dimensions.
    See :meth:`anndata.AnnData.raw` for more information.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, data):
        if data.data.raw is None:
            raise AttributeError(f"Raw data attribute doesn't exist \n{data}"
                                 f"If you wish to update raw, save raw first.")
        else:
            selected_genes = data.data.var_names
            self.logger.warning("RawData will change.")
            data.data.raw = data.data.raw[:, selected_genes].to_adata()


@register_preprocessor("misc")
class RemoveSplit(BaseTransform):
    """Remove a particular split from the data."""

    _DISPLAY_ATTRS = ("split_name", )

    def __init__(self, *, split_name: str, **kwargs):
        super().__init__(**kwargs)
        self.split_name = split_name

    def __call__(self, data):
        self.logger.info("Popping split: {self.split_name!r}")
        data.pop(split_name=self.split_name)


@register_preprocessor("misc")
class AlignMod(BaseTransform):
    """Aligning mods and metadata in multimodal data."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, data: Data) -> Data:
        mod1, mod2, meta1, meta2, test_sol = data.data.mod.values()
        meta1 = meta1[:, mod1.var.index]
        meta2 = meta2[:, mod2.var.index]
        test_sol = test_sol[:, mod1.var.index]
        data.data.mod["meta1"] = meta1
        data.data.mod["meta2"] = meta2
        data.data.mod["test_sol"] = test_sol
        return data
