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
        self.K = len(hidden_layers) + 1
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

    def predict(self, x: Vector) -> Vector:
        output = x
        activate = np.vectorize(activation)

        for layer in self.weights:
            output = np.append(output, 1.0)  # bias neuron
            summed = layer.dot(output)
            output = activate(summed)

        return output

    def fit(self, x, y, reps=1, beta=0.01):
        activate = np.vectorize(activation)
        err_hist = []
        dqdw = [None for _ in range(self.K + 1)]

        for iteration in range(reps):
            sub_err = []
            for i in range(len(y)):
                yk = y[i]
                xk = x[i, :]
                # feed forward
                s = [np.array([])]
                yi = [xk]
                output = xk
                for layer in self.weights:
                    output = np.append(output, 1.0)  # bias neuron
                    summed = layer.dot(output)
                    s.append(summed)
                    output = activate(summed)
                    yi.append(output)

                # backprop
                err = (output - yk) ** 2
                sub_err.append(err.sum())
                derr = 2 * (output - yk)

                dqdy = [np.array([]) for _ in range(self.K + 1)]
                dqds = [np.array([]) for _ in range(self.K + 1)]

                dqdy_next = derr
                dqds_next = dqdy_next * activation_derivative(s[self.K])

                dqdy[self.K] = dqdy_next
                dqds[self.K] = dqds_next

                for k in reversed(range(1, self.K)):
                    dqdy_now = np.transpose(self.weights[k]).dot(dqds_next)
                    dqdy_now = dqdy_now[:-1]  # drop bias neuron?
                    dqds_now = dqdy_now * activation_derivative(s[k])

                    dqdy[k] = dqdy_now
                    dqds[k] = dqds_now

                    dqdy_next, dqds_next = dqdy_now, dqds_now

                # derivatives in respect to weights
                for k in range(1, self.K + 1):
                    if dqdw[k] is None:
                        dqdw[k] = np.outer(dqds[k], np.transpose(yi[k - 1]))
                    else:
                        dqdw[k] += np.outer(dqds[k], np.transpose(yi[k - 1]))

            err_hist.append(sum(sub_err) / len(y))
            if (iteration + 1) % 100 == 0:
                print(f'Iteration {iteration + 1}: error {err_hist[-1]}')

            # update weights
            for k in range(1, self.K):
                # TODO bias neuron
                self.weights[k][:, :-1] = self.weights[k][:, :-1] - beta * dqdw[k + 1]

        return err_hist

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
