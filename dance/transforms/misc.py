from pprint import pformat

from dance.transforms.base import BaseTransform
from dance.typing import Any, Dict


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
