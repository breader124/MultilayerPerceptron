import numpy as np

Vector = np.ndarray
Matrix = np.ndarray


def activation_derivative(x: float) -> float:
    return x * (1.0 - x)


def activation(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def loss_function(y_target: float, y: float) -> float:
    return (y - y_target) ** 2
