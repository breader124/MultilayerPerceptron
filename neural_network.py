from typing import List
import numpy as np
from random import shuffle

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
        self.m = [
            np.zeros(layer.shape) for layer in self.weights
        ]
        self.v = [
            np.zeros(layer.shape) for layer in self.weights
        ]
        self.adam_t = 0

    def create_layers_matrices(self, layers: List[int]) -> List[Matrix]:
        matrices = []

        for i in range(len(layers) - 1):
            if i < len(layers) - 2:
                limit = 1 / np.sqrt(layers[i] + 1)
                matrix = np.random.random_sample((layers[i + 1], layers[i] + 1))
                # matrix = np.random.random_sample((layers[i + 1], layers[i]))
                matrix = matrix * 2 * limit - limit
            else:
                matrix = np.zeros((layers[i + 1], layers[i] + 1))
            matrices.append(matrix)

        return matrices

    def predict(self, x: Vector) -> Vector:
        output = x
        activate = np.vectorize(activation)

        for j, layer in enumerate(self.weights):
            output = np.append(output, 1.0)  # bias neuron
            summed = layer.dot(output)
            output = activate(summed)
            # if j == len(self.weights) - 1:
            #     tmp = np.exp(output)
            #     output = tmp / np.sum(tmp)

        return output

    def fit(self, x, y, reps=1, beta=0.01):
        activate = np.vectorize(activation)
        # err_hist = []

        for iteration in range(reps):
            dqdw = [None for _ in range(self.K + 1)]
            sub_err = []
            for i in range(len(y)):
                yk = y[i]
                xk = x[i, :]
                # feed forward
                s = [np.array([])]
                yi = [np.append(xk, 1.0)]
                # yi = [xk]
                output = xk
                for j, layer in enumerate(self.weights):
                    output = np.append(output, 1.0)  # bias neuron
                    summed = layer.dot(output)
                    s.append(summed)
                    output = activate(summed)

                    if j == len(self.weights) - 1:
                        yi.append(output)
                    else:
                        yi.append(np.append(output, 1.0))

                # backprop
                # err = -yk * np.log(output) - (1 - yk) * np.log(1 - output)
                err = np.mean(np.square(output - yk))
                sub_err.append(err)
                # derr = output - yk
                derr = 2 * (output - yk) / yk.size
                # derr = 2 * (output - yk)

                dqdy = [np.array([]) for _ in range(self.K + 1)]
                dqds = [np.array([]) for _ in range(self.K + 1)]

                dqdy_next = derr
                dqds_next = dqdy_next * activation_derivative(s[self.K])

                dqdy[self.K] = dqdy_next
                dqds[self.K] = dqds_next

                for k in reversed(range(1, self.K)):
                    dqdy_now = np.transpose(self.weights[k]).dot(dqds_next)
                    # dqdy_now = dqdy_now[:-1]  # drop bias neuron?
                    dqds_now = dqdy_now[:-1] * activation_derivative(s[k])
                    # dqds_now = dqdy_now * activation_derivative(s[k])

                    dqdy[k] = dqdy_now
                    dqds[k] = dqds_now

                    dqdy_next, dqds_next = dqdy_now, dqds_now

                # derivatives in respect to weights
                for k in range(1, self.K + 1):
                    if dqdw[k] is None:
                        dqdw[k] = np.outer(dqds[k], yi[k - 1]) / len(y)
                        # dqdw[k] = np.outer(dqds[k], np.append(yi[k - 1], 1.0))
                    else:
                        dqdw[k] += np.outer(dqds[k], yi[k - 1]) / len(y)

            # update weights
            self._update_weights(beta, dqdw)

        return sum(sub_err) / len(sub_err)

    def _update_weights(self, beta, dqdw):
        optim = 'adam'
        if optim == 'sgd':
            self._sgd(beta, dqdw)
        elif optim == 'lm':
            self._levenberg(beta, dqdw)
        elif optim == 'adam':
            self._adam(beta, dqdw)

    def _sgd(self, beta, dqdw):
        for k in range(0, self.K):
            # TODO bias neuron
            # self.weights[k][:, :-1] = self.weights[k][:, :-1] - beta * dqdw[k + 1]
            self.weights[k] = self.weights[k] - beta * dqdw[k + 1]

    def _levenberg(self, beta, dqdw):
        raise Exception('Not implemented')
        th = np.array([])
        for k in range(0, self.K):
            vectorized = self.weights[k].reshape((1, self.weights[k].size))
            th = np.append(th, vectorized)

        dth = np.array([])
        for k in range(0, self.K):
            vectorized = dqdw[k + 1].reshape((1, dqdw[k + 1].size))
            dth = np.append(dth, vectorized)

    def _adam(self, beta, dqdw):
        dqdw = dqdw[1:]
        eps = 1e-7
        b1, b2 = 0.9, 0.999

        self.m = [
            b1 * tmp + (1 - b1) * dqdw[i]
            for i, tmp in enumerate(self.m)
        ]
        self.v = [
            b2 * tmp + (1 - b2) * np.square(dqdw[i])
            for i, tmp in enumerate(self.v)
        ]

        md = [tmp / (1 - b1**self.adam_t) for tmp in self.m]
        vd = [tmp / (1 - b2**self.adam_t) for tmp in self.v]

        self.weights = [
            tmp - beta * md[i] / np.sqrt(vd[i] + eps)
            for i, tmp in enumerate(self.weights)
        ]

        self.adam_t += 1

    def batch_fit(self, X, y, batch=10, reps=100, beta=0.1):
        self.adam_t = 1
        err_hist = []
        parts = int(np.ceil(len(y) / batch))

        for iteration in range(reps):
            p = np.random.permutation(len(y))
            Xs, ys = X[p], y[p]
            err = 0
            for i in range(parts):
                start = batch * i
                stop = min(start + batch, len(y))
                Xt, yt = Xs[start:stop], y[start:stop]
                err += self.fit(Xt, yt, reps=1, beta=beta)

            err_hist.append(err / parts)
            # err_hist.append(self.eval(X, y))
            if (iteration + 1) % 100 == 0:
                # beta = beta * 0.7
                print(f'Iteration {iteration + 1}: loss: {err_hist[-1]}, accuracy: {self.eval(X, y):.4f}')

        return err_hist

    def eval(self, X, y):
        alle = np.array([self.predict(gle) for gle in X])
        maxed = np.argmax(alle, axis=1)
        y_raw = np.argmax(y, axis=1)
        errs = [a != b for a, b in zip(y_raw, maxed)]
        return 1 - (sum(errs) / len(y))

    def fit_test(self, x, y, beta=0.01):
        activate = np.vectorize(activation)

        dqdw = [None for _ in range(self.K + 1)]
        sub_err = []
        yk = y
        xk = x
        # feed forward
        s = [np.array([])]
        yi = [np.append(xk, 1.0)]
        # yi = [xk]
        output = xk
        for layer in self.weights:
            output = np.append(output, 1.0)  # bias neuron
            summed = layer.dot(output)
            s.append(summed)
            output = activate(summed)
            yi.append(np.append(output, 1.0))
            # yi.append(output)
        yi[-1] = output

        # backprop
        err = 0.5 * (output - yk) ** 2
        sub_err.append(err.sum())
        derr = (output - yk)

        dqdy = [np.array([]) for _ in range(self.K + 1)]
        dqds = [np.array([]) for _ in range(self.K + 1)]

        dqdy_next = derr
        dqds_next = dqdy_next * activation_derivative(s[self.K])

        dqdy[self.K] = dqdy_next
        dqds[self.K] = dqds_next

        for k in reversed(range(1, self.K)):
            dqdy_now = np.transpose(self.weights[k]).dot(dqds_next)
            # dqdy_now = dqdy_now[:-1]  # drop bias neuron?
            dqds_now = dqdy_now[:-1] * activation_derivative(s[k])
            # dqds_now = dqdy_now * activation_derivative(s[k])

            dqdy[k] = dqdy_now
            dqds[k] = dqds_now

            dqdy_next, dqds_next = dqdy_now, dqds_now

        # derivatives in respect to weights
        for k in range(1, self.K + 1):
            if dqdw[k] is None:
                dqdw[k] = np.outer(dqds[k], yi[k - 1])
                # dqdw[k] = np.outer(dqds[k], np.append(yi[k - 1], 1.0))
            else:
                dqdw[k] += np.outer(dqds[k], yi[k - 1])

        # update weights
        for k in range(0, self.K):
            # TODO bias neuron
            # self.weights[k][:, :-1] = self.weights[k][:, :-1] - beta * dqdw[k + 1]
            self.weights[k] = self.weights[k] - beta * dqdw[k + 1]

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
