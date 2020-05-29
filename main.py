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
    return X, y


if __name__ == '__main__':
    seed(42)
    np.random.seed(42)

    X, y = datasets.load_iris(return_X_y=True)
    X, y = clean_data(X, y)

    model = NeuralNetwork(2, [7], 2)
    in_ = np.array([1, 1])
    exp = np.array([1, 0])
    out = model.predict(in_)
    print(out)

    errs = model.fit(in_, exp, beta=0.1)
    plt.plot(errs)
    plt.legend(['Error'])
    plt.show()

    out = model.predict(in_)
    print(out)
