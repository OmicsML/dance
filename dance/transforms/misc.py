from pprint import pformat

from dance.transforms.base import BaseTransform
from dance.typing import Any, Dict, Tuple


class Compose(BaseTransform):
    """Compose transformation by combining several transfomration objects."""

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

    def __call__(self, data):
        self.logger.info(f"Applying composed transformations:\n{self!r}")
        for transform in self.transforms:
            transform(data)


class SetConfig(BaseTransform):
    """Set configuration options of a dance data object."""

    _DISPLAY_ATTRS = ("config_dict", )

    def __init__(self, config_dict: Dict[str, Any], **kwargs):
        """Initialize SetConfig object.

        Parameters
        ----------
        config_dict
            Dance data object configuration dictionary. See :meth:`~dance.data.base.BaseData.set_config_from_dict`.

        """
        super().__init__(**kwargs)
        self.config_dict = config_dict

    def __call__(self, data):
        self.logger.info(f"Updating the dance data object config options:\n{pformat(self.config_dict)}")
        data.set_config_from_dict(self.config_dict)
