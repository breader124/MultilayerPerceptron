from typing import List
import numpy as np

from math_utils import activation_derivative, activation, \
    Matrix, Vector


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
            limit = 1 / np.sqrt(layers[i] + 1)

            matrix = np.random.random_sample((layers[i + 1], layers[i] + 1))
            matrix = matrix * 2 * limit - limit
            matrices.append(matrix)

        return matrices

    def predict(self, input_data: Vector) -> Vector:
        output = input_data
        activate = np.vectorize(activation)

        for layer in self.weights:
            output = np.append(output, 1.0)  # bias neuron
            summed = layer.dot(output)
            output = activate(summed)

        return output

    def debug_compute(self, input_data: Vector):
        output_matrix = []
        sum_matrix = []
        for layer in self.weights:
            layer_output = []
            layer_sums = []
            for neuron in layer:
                summed = neuron[-1]
                for i, weight in enumerate(neuron[:-1]):
                    summed += input_data[i] * weight
                activated = activation(summed)

                layer_sums.append(summed)
                layer_output.append(activated)

            input_data = layer_output
            sum_matrix.append(layer_sums)
            output_matrix.append(layer_output)

        return output_matrix, sum_matrix

    def backward_propagation(self, output_matrix, sum_matrix,
                             target_results, input_values):
        derivatives = []

        last_layer_der = []
        for out, s, ideal in zip(output_matrix[-1], sum_matrix[-1],
                                 target_results):
            y_derivative = 2 * (out - ideal)
            sum_derivative = y_derivative * activation_derivative(s)
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
                sum_derivative = y_derivative * activation_derivative(
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
