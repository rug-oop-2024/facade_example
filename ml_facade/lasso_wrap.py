import numpy as np
from sklearn.linear_model import Lasso as ScikitLearnLasso

from ml_facade.model import Model

class Lasso(Model):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self._wrapped_model = ScikitLearnLasso(*args, **kwargs)
        self._hyperparameters = {
            param: value
            for param, value in self._wrapped_model.get_params().items()
            if param not in ("coef_", "intercept_")
        }
    
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        self._wrapped_model.fit(observations, ground_truth)
        self._parameters = {
            param: value
            for param, value in self._wrapped_model.get_params().items()
            if param in ("coef_", "intercept_")
        }
    
    def predict(self, observations:np.ndarray) -> np.ndarray:
        return self._wrapped_model.predict(observations)
    

