import argparse
from matplotlib import pyplot as plt
import numpy as np
from random import seed
import seaborn as sns
from sklearn import datasets

sns.set()

from neural_network import NeuralNetwork


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('layers', metavar='N', type=int, nargs='+')
    parser.add_argument('--inp', type=int)

    args = parser.parse_args()
    return args.inp, args.layers


def clean_data(X, y):
    # TODO normalize X
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

    # X, y = X[list(range(10)) + list(range(120, 130))], y[list(range(10)) + list(range(120, 130))]

    inputs = X.shape[1]
    outputs = y.shape[1]

    model = NeuralNetwork(inputs, [20, 20], outputs)
    in_ = X[0, :]
    exp = y[0]
    out = model.predict(in_)
    print(out)

    # errs = model.fit(X, y, reps=500, beta=0.5)
    errs = model.batch_fit(X, y, batch=15, reps=2000, beta=0.005)
    plt.plot(errs)
    plt.legend(['Loss'])
    plt.show()

    out = model.predict(in_)
    print(out)

    acc = model.eval(X, y)
    print(f'Samples classified to correct class: {int(acc * len(y))}')
