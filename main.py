import argparse

from neural_network import NeuralNetwork


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('layers', metavar='N', type=int, nargs='+')
    parser.add_argument('--inp', type=int)

    args = parser.parse_args()
    return args.inp, args.layers


if __name__ == '__main__':
    model = NeuralNetwork(2, [3, 4], 3)
    print('Hello World :)')
