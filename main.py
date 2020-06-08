import argparse
import numpy as np
import seaborn as sns

from random import seed
from sklearn import datasets
from matplotlib import pyplot as plt
from neural_network import cross_validation

sns.set()

from neural_network import NeuralNetworkStructure


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('layers', metavar='N', type=int, nargs='+')
    parser.add_argument('--inp', type=int)

    args = parser.parse_args()
    return args.inp, args.layers


def clean_data(X, y):
    X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    X = X * 2 - 1
    y = np.array(list(
        [int(tmp == 0), int(tmp == 1), int(tmp == 2)] for tmp in y
    ))
    return X, y


if __name__ == '__main__':
    seed(42)
    np.random.seed(42)

    X, y_raw = datasets.load_iris(return_X_y=True)
    X, y = clean_data(X, y_raw)

    inputs = X.shape[1]
    outputs = y.shape[1]

    model_structure = NeuralNetworkStructure(inputs, [50, 50], outputs)
    model, mean_error, training_errs, val_errs = cross_validation(X, y, model_structure, 3)
    print(f'Mean model loss: {mean_error}')

    plt.plot(training_errs)
    plt.plot(val_errs)
    plt.legend(['Loss', 'Val Loss'])
    plt.show()

    # out = model.predict(in_)
    # print(out)

    acc = model.eval(X, y)
    print(f'Samples classified to correct class: {int(acc * len(y))}')
