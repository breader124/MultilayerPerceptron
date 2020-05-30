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
    # TODO normalize X, change y from {0,1,2} to {(0,0,1), (0,1,0), (1,0,0)}
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

    model = NeuralNetwork(inputs, [7], outputs)
    in_ = X[0, :]
    exp = y[0]
    out = model.predict(in_)
    print(out)

    errs = model.fit(X, y, reps=500, beta=0.1)
    plt.plot(errs)
    plt.legend(['Error'])
    plt.show()

    out = model.predict(in_)
    print(out)

    alle = np.array([model.predict(gle) for gle in X])
    maxed = np.argmax(alle, axis=1)
    errs = [a != b for a, b in zip(y_raw, maxed)]
    print(f'Samples classified to wrong class: {sum(errs)}')
