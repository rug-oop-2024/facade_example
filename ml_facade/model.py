from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
from copy import deepcopy



class ParametersDict(dict):
    def _get_keys_as_list(self) -> list:
        return list(self.keys()).sort()
    
    def update(self, new_dict: "ParametersDict") -> None:
        if not isinstance(new_dict, "ParametersDict"):
            new_dict = ParametersDict(new_dict)
        if self._get_keys_as_list() == new_dict._get_keys_as_list() or len(self) == 0:
            super().update(new_dict)
        else:
            raise AttributeError(f"ParametersDict.update expects another ParametersDict with the same keys. Found {self.keys()} vs {new_dict.keys()}")

class Model(ABC):
    _parameters: ParametersDict = ParametersDict({})
    _hyperparameters: ParametersDict = ParametersDict({})

    @abstractmethod
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, observations: np.ndarray) -> np.ndarray:
        pass

    @property
    def parameters(self) -> ParametersDict:
        return deepcopy(self._parameters)
    
    @parameters.setter
    def parameters(self, new_params: ParametersDict) -> None:
        self._parameters.update(new_params)

    @property
    def hyperparameters(self) -> ParametersDict:
        return deepcopy(self._hyperparameters)
    
    @hyperparameters.setter
    def hyperparameters(self, new_hyperparams: ParametersDict) -> None:
        self._hyperparameters.update(new_hyperparams)