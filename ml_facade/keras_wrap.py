import keras
from sklearn.linear_model import Lasso as ScikitLearnLasso
import numpy as np
from typing import Callable

from ml_facade.model import Model

class NeuralNetworkKerasWrap(Model):
    def __init__(
            self,
            layers_size: list[int],
            input_shape: list[int],
            hidden_layer_activation: Callable[[np.ndarray], np.ndarray] = keras.activations.relu,
            final_layer_activation: Callable[[np.ndarray], np.ndarray] = keras.activations.softmax,
            flatten_input: bool = False,
        ) -> None:
        super().__init__()
        self._wrapped_model = self._build_keras_model(
            layers_size, input_shape, hidden_layer_activation, final_layer_activation, flatten_input
        )
        self._parameters = self._wrapped_model.get_weights()

    def _build_keras_model(
            self,
            layers_size: list[int],
            input_shape: list[int],
            hidden_layer_activation: Callable[[np.ndarray], np.ndarray] = keras.activations.relu,
            final_layer_activation: Callable[[np.ndarray], np.ndarray] = keras.activations.softmax,
            flatten_input: bool = False
        ) -> keras.Model:
        layers = []
        if flatten_input:
            layers.append(keras.layers.Flatten())
        for layer_id, layer_size in enumerate(layers_size):
            activation = hidden_layer_activation if layer_id <= len(layers_size) else final_layer_activation
            layers.append(keras.layers.Dense(layer_size, activation=activation))
        model = keras.Sequential(layers)
        model.build(input_shape=input_shape)
        return model
    
    def _compile_model(
            self,
            optimizer_cls: type = keras.optimizers.Adam,
            loss_cls: type = keras.losses.BinaryCrossentropy
        ) -> None:
        optimizer = optimizer_cls()
        loss = loss_cls()
        self._wrapped_model.compile(
            optimizer = optimizer,
            loss = loss
        )
        self._hyperparameters = {
            "optimizer": optimizer,
            "loss": loss
        }
    
    def fit(
            self,
            observations: np.ndarray,
            ground_truth: np.ndarray,
            num_epochs: int,
            optimizer: type = keras.optimizers.Adam,
            loss: type = keras.losses.BinaryCrossentropy
        ) -> None:
        self._compile_model(optimizer, loss)
        self._wrapped_model.fit(observations, ground_truth, epochs=num_epochs)

    def predict(
            self,
            observations: np.ndarray
        ):
        self._wrapped_model.predict(observations)
                
            