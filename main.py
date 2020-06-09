import argparse
import numpy as np
import seaborn as sns

from random import seed
from matplotlib import pyplot as plt
from sklearn import datasets
from neural_network import NeuralNetworkStructure
from neural_network import cross_validation

sns.set()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('layers', type=int)
    parser.add_argument('min', type=int)
    parser.add_argument('max', type=int)
    parser.add_argument('step', type=int)
    parser.add_argument('--algorithm', type=str, default='adam')
    parser.add_argument('--beta', type=float, default=0.001, help='Learning rate')

    args = parser.parse_args()
    return args.layers, args.min, args.max, args.step, args.algorithm, args.beta


def clean_data(X, y):
    X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    X = X * 2 - 1
    y = np.array(list(
        [int(tmp == 0), int(tmp == 1), int(tmp == 2)] for tmp in y
    ))
    return X, y


if __name__ == '__main__':
    layers_num, min_neurons, max_neurons, step, algorithm, beta = parse_args()

    seed(42)
    np.random.seed(42)

    X, y_raw = datasets.load_iris(return_X_y=True)
    X, y = clean_data(X, y_raw)

    inputs = X.shape[1]
    outputs = y.shape[1]

    k_fold = 5
    legend = []
    for neurons_num in range(min_neurons, max_neurons + 1, step):
        hidden_layers = [neurons_num] * layers_num
        model_structure = NeuralNetworkStructure(inputs, hidden_layers, outputs)
        model, errors = cross_validation(X, y, model_structure, k_fold, algorithm, beta)

        legend = legend + [neurons_num]
        plt.plot(errors)

        print(f'Finished for {neurons_num} neurons')

    plt.legend(legend)
    plt.show()
