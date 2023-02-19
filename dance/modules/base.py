import os
from abc import ABC, abstractmethod, abstractstaticmethod
from time import time

import torch

from dance import logger
from dance.data import Data
from dance.transforms.base import BaseTransform
from dance.typing import Any, Mapping, Optional, Tuple, Union
from dance.utils.metrics import resolve_score_func


class BaseMethod(ABC):

    _DEFAULT_METRIC: Optional[str] = None
    _DISPLAY_ATTRS: Tuple[str] = ()

    def __repr__(self) -> str:
        display_attrs_str_list = [f"{i}={getattr(self, i)!r}" for i in self._DISPLAY_ATTRS]
        display_attrs_str = ", ".join(display_attrs_str_list)
        return f"{self.name}({display_attrs_str})"

    def preprocess(self, data: Data, /, **kwargs):
        self.preprocessing_pipeline(**kwargs)(data)

    @abstractstaticmethod
    def preprocessing_pipeline(**kwargs) -> BaseTransform:
        ...

    @abstractmethod
    def fit(self, x, y, **kwargs):
        ...

    def predict_proba(self, x):
        raise NotImplementedError

    @abstractmethod
    def predict(self, x):
        ...

    def fit_predict(self, x, y, **fit_kwargs):
        self.fit(x, y, **fit_kwargs)
        pred = self.predict(x)
        return pred

    def score(self, x, y, score_func: Optional[Union[str, Mapping[Any, float]]] = None,
              return_pred: bool = False) -> Union[float, Tuple[float, Any]]:
        y_pred = self.predict(x)
        func = resolve_score_func(score_func or self._DEFAULT_METRIC)
        score = func(y, y_pred)
        return (score, y_pred) if return_pred else score


class BasePretrain(ABC):

    @property
    def is_pretrained(self) -> bool:
        return getattr(self, "_is_pretrained", False)

    def _pretrain(self, *args, force_pretrain: bool = False, **kwargs):
        pt_path = getattr(self, "pretrain_path", None)
        if not force_pretrain:
            if self.is_pretrained:
                logger.info("Skipping pre_train as the model appears to be pretrained already. "
                            "If you wish to force pre-training, please set 'force_pretrain' to True.")
                return

            if pt_path is not None and os.path.isfile(pt_path):
                logger.info(f"Loading pre-trained model from {pt_path}")
                self.load_pretrained(pt_path)
                self._is_pretrained = True
                return

        logger.info("Pre-training started")
        if pt_path is None:
            logger.warning("`pretrain_path` is not set, pre-trained model will not be saved.")
        else:
            logger.info(f"Pre-trained model will to saved to {pt_path}")

        t = time()
        self.pretrain(*args, **kwargs)
        elapsed = time() - t
        logger.info(f"Pre-training finished (took {elapsed:.2f} seconds)")
        self._is_pretrained = True

        if pt_path is not None:
            logger.info(f"Saving pre-trained model to {pt_path}")
            self.save_pretrained(pt_path)

    def pretrain(self, *args, **kwargs):
        ...

    def save_pretrained(self, path, **kwargs):
        ...

    def load_pretrained(self, path, **kwargs):
        ...


class TorchNNPretrain(BasePretrain, ABC):

    def save_pretrained(self, path):
        torch.save(self.state_dict(), path)

    def load_pretrained(self, path):
        device = getattr(self, "device", None)
        # FIX: check if map_loc can be set to `cuda`, or is the index strictly required
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint)


class BaseClassificationMethod(BaseMethod):

    _DEFAULT_METRIC = "acc"


class BaseRegressionMethod(BaseMethod):

    _DEFAULT_METRIC = "rmse"


class BaseClusteringMethod(BaseMethod):

    _DEFAULT_METRIC = "ari"
