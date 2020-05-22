import argparse
from math import atan, pi, sqrt
from random import uniform
from typing import List

from math_utils import arctan_derivative, scaled_atan, matrix_dot_vector

Matrix = List[List[float]]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('layers', metavar='N', type=int, nargs='+')
    parser.add_argument('--inp', type=int)

    args = parser.parse_args()
    return args.inp, args.layers


class NeuralNetwork:
    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int],
        output_size: int
    ):
        self.weights = self.create_layers_matrices(
            [input_size] + hidden_layers + [output_size]
        )

    def create_layers_matrices(self, layers: List[int]) -> List[Matrix]:
        matrices = []

        for i in range(len(layers) - 1):
            def init_weight():
                return uniform(-1 / sqrt(layers[i] + 1),
                               1 / sqrt(layers[i] + 1))

            matrix = [
                [init_weight() for _ in range(layers[i] + 1)]
                for _ in range(layers[i + 1])
            ]
            matrices.append(matrix)

        return matrices

    def compute(self, input_data):
        output_matrix = []
        sum_matrix = []
        for layer in self.weights:
            layer_output = []
            layer_sums = []
            for neuron in layer:
                summed = neuron[-1]
                for i, weight in enumerate(neuron[:-1]):
                    summed += input_data[i] * weight
                activated = scaled_atan(summed)

                layer_sums.append(summed)
                layer_output.append(activated)

            input_data = layer_output
            sum_matrix.append(layer_sums)
            output_matrix.append(layer_output)

        return output_matrix, sum_matrix

    def loss_function(self, given_y, y):
        return (y - given_y) ** 2

    def backward_propagation(self, output_matrix, sum_matrix,
                             given_results, input_values):
        derivatives = []

        last_layer_der = []
        for out, s, ideal in zip(output_matrix[-1], sum_matrix[-1],
                                 given_results):
            y_derivative = 2 * (out - ideal)
            sum_derivative = y_derivative * arctan_derivative(s)
            res = {
                'y': y_derivative,
                's': sum_derivative
            }
            last_layer_der.append(res)
        derivatives.append(last_layer_der)

        for layer_index in reversed(range(len(self.weights) - 1)):
            new_layer = []
            for neuron_index in range(len(self.weights[layer_index])):
                y_derivative = 0
                for neuron_next_layer in range(
                    len(self.weights[layer_index + 1])):
                    neuron_sum = derivatives[-1][neuron_next_layer]['s']
                    weight = self.weights[layer_index + 1][neuron_next_layer][
                        neuron_index]
                    y_derivative += neuron_sum * weight
                sum_derivative = y_derivative * arctan_derivative(
                    sum_matrix[layer_index][neuron_index])
                res = {
                    'y': y_derivative,
                    's': sum_derivative
                }
                new_layer.append(res)
            derivatives.append(new_layer)

        first_layer_der = []
        for input_value_index in range(len(input_values)):
            y_derivative = 0
            for neuron_first_layer in range(len(self.weights[1])):
                y_derivative += derivatives[-1][neuron_first_layer]['s'] * \
                                self.weights[1][neuron_first_layer][
                                    input_value_index]
            first_layer_der.append(y_derivative)

        return derivatives, first_layer_der

    def train(self):
        pass


if __name__ == '__main__':
    print("Hello World :)")
