from typing import List
import numpy as np

from math_utils import activation_derivative, activation, Matrix, Vector


def create_layers_matrices(layers: List[int]) -> List[Matrix]:
    matrices = []

    for i in range(len(layers) - 1):
        limit = 1 / np.sqrt(layers[i] + 1)

        if i < len(layers) - 2:
            matrix = np.random.random_sample((layers[i + 1], layers[i] + 1))
        else:
            matrix = np.random.random_sample((layers[i + 1], layers[i] + 1))

        matrix = matrix * 2 * limit - limit
        matrices.append(matrix)

    return matrices


def cross_validation(X, y, neural_network_struct, k, algorithm, beta):
    z = list(zip(X, y))
    np.random.shuffle(z)
    shuffled = np.array(z)

    batches = np.array_split(shuffled, k)
    batches_repeats = list()
    for validation_set in batches:
        training_set = [x for x in batches if x != validation_set]
        training_set = np.vstack(training_set)

        model = NeuralNetwork(neural_network_struct)
        repeats = model.early_stop_fit(training_set, validation_set, algorithm=algorithm, beta=beta)
        batches_repeats.append(repeats)

    model = NeuralNetwork(neural_network_struct)
    mean_repeats = int(np.mean(batches_repeats))
    errs = model.fixed_repeats_fit(shuffled, algorithm=algorithm, repeats=mean_repeats)

    return model, errs


def split_dataset(dataset):
    return np.stack(dataset[:, 0]), np.stack(dataset[:, 1])


def check_increasing(hist_err):
    mean = np.mean(hist_err)
    return hist_err[-1] > mean


class NeuralNetworkStructure:
    def __init__(
            self,
            input_size: int,
            hidden_layers: List[int],
            output_size: int
    ):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size

    def full_structure(self):
        return [self.input_size] + self.hidden_layers + [self.output_size]


class NeuralNetwork:
    def __init__(
            self,
            structure: NeuralNetworkStructure
    ):
        self.K = len(structure.hidden_layers) + 1
        self.weights = create_layers_matrices(structure.full_structure())

        self.m = [
            np.zeros(layer.shape) for layer in self.weights
        ]
        self.v = [
            np.zeros(layer.shape) for layer in self.weights
        ]
        self.adam_t = 0

    def predict(self, x: Vector) -> Vector:
        output = x

        for j, layer in enumerate(self.weights):
            output = np.append(output, 1.0)  # bias neuron
            summed = layer.dot(output)
            output = activation(summed)

        return output

    def early_stop_fit(self, training_set, validation_set, batch=10, algorithm='adam', beta=0.001):
        X, y = split_dataset(training_set)
        X_val, y_val = split_dataset(validation_set)
        self.adam_t = 1

        err_hist = []
        val_err_hist = []
        parts = int(np.ceil(len(y) / batch))

        repeats = 1
        while True:
            err = self.single_batch_fit(X, y, parts, batch, algorithm, beta)
            err_hist.append(err / parts)
            val_err_hist.append(self.error(X_val, y_val))

            if len(val_err_hist) > 30:
                if check_increasing(val_err_hist[-30:-1]):
                    break
            repeats = repeats + 1

        return repeats

    def fixed_repeats_fit(self, dataset, repeats, algorithm, batch=10, beta=0.001):
        X, y = split_dataset(dataset)
        self.adam_t = 1

        err_hist = []
        parts = int(np.ceil(len(y) / batch))

        for i in range(repeats):
            err = self.single_batch_fit(X, y, parts, batch, algorithm, beta)
            err_hist.append(err / parts)

        return err_hist

    def single_batch_fit(self, X, y, parts, batch, algorithm, beta):
        p = np.random.permutation(len(y))
        Xs, ys = X[p], y[p]
        err = 0
        for i in range(parts):
            start = batch * i
            stop = min(start + batch, len(y))
            Xt, yt = Xs[start:stop], ys[start:stop]
            err += self.fit(Xt, yt, algorithm=algorithm, beta=beta)

        return err

    def fit(self, x, y, algorithm='adam', beta=0.01):
        dqdw = [np.zeros(layer.shape) for layer in self.weights]
        sub_err = []

        for i in range(len(y)):
            yk = y[i]
            xk = x[i, :]

            output, s = self.feed_forward(xk)

            # backward propagation
            err = np.mean(np.square(output - yk))
            sub_err.append(err)
            delta = output - yk
            dqdw[-1] += np.outer(delta, s[-1])
            for k in reversed(range(1, len(dqdw))):
                delta = self.weights[k].T.dot(delta) * activation_derivative(s[k])
                delta = delta[:-1]
                dqdw[k - 1] += np.outer(delta, s[k - 1])

        dqdw = [dqdw[i] / len(x) for i in range(len(dqdw))]
        self._update_weights(algorithm, beta, dqdw)

        return sum(sub_err) / len(sub_err)

    def feed_forward(self, xk):
        output = xk
        s = []
        for layer in self.weights:
            output = np.append(output, 1.0)  # bias neuron
            s.append(output)
            summed = layer.dot(output)
            output = activation(summed)
        return output, s

    def error(self, x, y):
        errors = []
        for i in range(len(y)):
            xk = x[i, :]
            yk = y[i]
            output, _ = self.feed_forward(xk)
            err = np.mean(np.square(output - yk))
            errors.append(err)

        return np.mean(errors)

    def eval(self, X, y):
        alle = np.array([self.predict(gle) for gle in X])
        maxed = np.argmax(alle, axis=1)
        y_raw = np.argmax(y, axis=1)
        errs = [a != b for a, b in zip(y_raw, maxed)]
        return 1 - (sum(errs) / len(y))

    def _update_weights(self, algorithm, beta, dqdw):
        if algorithm == 'sgd':
            self._sgd(beta, dqdw)
        elif algorithm == 'adam':
            self._adam(beta, dqdw)
        else:
            raise NotImplementedError(f'Algorithm {algorithm} not implemented!')

    def _sgd(self, beta, dqdw):
        for k in range(len(self.weights)):
            self.weights[k] -= beta * dqdw[k]

    def _adam(self, beta, dqdw):
        eps = 1e-7
        b1, b2 = 0.9, 0.999

        self.m = [
            b1 * tmp + (1.0 - b1) * dqdw[i]
            for i, tmp in enumerate(self.m)
        ]
        self.v = [
            b2 * tmp + (1.0 - b2) * np.square(dqdw[i])
            for i, tmp in enumerate(self.v)
        ]

        md = [tmp / (1 - b1 ** self.adam_t) for tmp in self.m]
        vd = [tmp / (1 - b2 ** self.adam_t) for tmp in self.v]

        self.weights = [
            tmp - beta * md[i] / np.sqrt(vd[i] + eps)
            for i, tmp in enumerate(self.weights)
        ]

        self.adam_t += 1
