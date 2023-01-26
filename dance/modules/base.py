from abc import ABC, abstractmethod, abstractstaticmethod

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


class BaseClassificationMethod(BaseMethod):

    _DEFAULT_METRIC = "acc"


class BaseRegressionMethod(BaseMethod):

    _DEFAULT_METRIC = "rmse"


class BaseClusteringMethod(BaseMethod):

    _DEFAULT_METRIC = "ari"
