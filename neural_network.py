import argparse
import math


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('layers', metavar='N', type=int, nargs='+')
    parser.add_argument('--inp', type=int)

    args = parser.parse_args()
    return args.inp, args.layers


def create_layers_matrices(inp, layers):
    matrices = list()

    num_of_neurons = [inp] + layers

    for i in range(len(num_of_neurons) - 1):
        matrix = list()
        row = ([1] * (num_of_neurons[i] + 1))
        for j in range(num_of_neurons[i + 1]):
            matrix.append(row)
        matrices.append(matrix)

    return matrices


def compute():
    pass


if __name__ == '__main__':
    input_num, neurons_in_layers = parse_args()
    weight_matrices = create_layers_matrices(input_num, neurons_in_layers)
    for m in weight_matrices:
        print(m)
