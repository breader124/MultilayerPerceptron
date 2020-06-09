import numpy as np

Vector = np.ndarray
Matrix = np.ndarray


def activation_derivative(x):
    return x * (1.0 - x)


def activation(x):
    return 1 / (1 + np.exp(-x))
