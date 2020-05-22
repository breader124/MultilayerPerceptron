from math import atan, pi


def arctan_derivative(x):
    return 1 / (1 + x ** 2)


def scaled_atan(x):
    value = atan(x)
    offset = pi / 2
    return (value + offset) / pi


def matrix_dot_vector(A, b):
    result = [
        sum(x * y for x, y in zip(row, b))
        for row in A
    ]

    return result


def loss_function(y_target, y):
    return (y - y_target) ** 2
