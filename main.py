from tensorflow.keras.datasets import mnist
import numpy as np
from keras.losses import CategoricalCrossentropy
from keras.utils import to_categorical
from ml_facade import keras_wrap

def normalize_data(data: np.ndarray, upper_bound: float, lower_bound: float):
    assert upper_bound >= lower_bound, f"Upper bound {upper_bound} is < than lower bound {lower_bound}"
    return (data - lower_bound) / (upper_bound - lower_bound)


if __name__ == "__main__":
    (train_obs, train_ground), (test_obs, test_ground) = mnist.load_data()
    model = keras_wrap.NeuralNetworkKerasWrap(
        input_shape = [None, 28, 28],
        layers_size = [16, 32, 64, 10],
        flatten_input = True
    )

    train_obs = normalize_data(train_obs, 255, 0)
    test_obs = normalize_data(test_obs, 255, 0)

    train_ground = to_categorical(train_ground)
    test_ground = to_categorical(test_ground)

    # this should not be done!
    print(model._wrapped_model.summary())

    model.fit(train_obs, train_ground, num_epochs=2, loss=CategoricalCrossentropy)
    predictions = model.predict(test_obs)

    print(model.parameters)
    print(model.hyperparameters)

