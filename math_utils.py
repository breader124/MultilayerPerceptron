from math import exp
from typing import List

Vector = List[float]
Matrix = List[List[float]]


def activation_derivative(x: float) -> float:
    return activation(x) * (1 - activation(x))


def activation(x: float) -> float:
    return 1 / (1 + exp(-x))


def matrix_dot_vector(A: Matrix, b: Vector) -> Vector:
    result = [
        sum(x * y for x, y in zip(row, b))
        for row in A
    ]

    return result


def loss_function(y_target: float, y: float) -> float:
    return (y - y_target) ** 2
