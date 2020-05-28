import argparse
from random import seed
import numpy as np

from neural_network import NeuralNetwork


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('layers', metavar='N', type=int, nargs='+')
    parser.add_argument('--inp', type=int)

    args = parser.parse_args()
    return args.inp, args.layers


if __name__ == '__main__':
    seed(42)
    model = NeuralNetwork(2, [3, 2], 2)
    in_ = np.array([1, 1])
    out = model.predict(in_)
    out2 = model.debug_compute(in_)
    print(f'{out}')
    print(out2)
