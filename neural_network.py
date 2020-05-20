import argparse
from math import atan, pi


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


def compute(input_data, weights):
    output_matrix = list()
    sum_matrix = list()
    for layer in weights:
        layer_output = list()
        layer_sums = list()
        for neuron in layer:
            summed = neuron[-1]
            for i, weight in enumerate(neuron[:-1]):
                summed += input_data[i] * weight
            activated = scaled_atan(summed)

            layer_sums.append(summed)
            layer_output.append(activated)

        input_data = layer_output
        sum_matrix.append(layer_sums)
        output_matrix.append(layer_output)

    return output_matrix, sum_matrix


def scaled_atan(x):
    value = atan(x)
    offset = pi / 2
    return (value + offset) / pi


def loss_function(given_y, y):
    return (y - given_y) ** 2


def backward_propagation(output_matrix, sum_matrix, weights, given_results, input_values):
    derivatives = list()

    last_layer_der = list()
    for out, s, ideal in zip(output_matrix[-1], sum_matrix[-1], given_results):
        y_derivative = 2 * (out - ideal)
        sum_derivative = y_derivative * arctan_derivative(s)
        res = {
            'y': y_derivative,
            's': sum_derivative
        }
        last_layer_der.append(res)
    derivatives.append(last_layer_der)

    for layer_index in reversed(range(len(weights) - 1)):
        new_layer = []
        for neuron_index in range(len(weights[layer_index])):
            y_derivative = 0
            for neuron_next_layer in range(len(weights[layer_index + 1])):
                neuron_sum = derivatives[-1][neuron_next_layer]['s']
                weight = weights[layer_index + 1][neuron_next_layer][neuron_index]
                y_derivative += neuron_sum * weight
            sum_derivative = y_derivative * arctan_derivative(sum_matrix[layer_index][neuron_index])
            res = {
                'y': y_derivative,
                's': sum_derivative
            }
            new_layer.append(res)
        derivatives.append(new_layer)

    first_layer_der = list()
    for input_value_index in range(len(input_values)):
        y_derivative = 0
        for neuron_first_layer in range(len(weights[1])):
            y_derivative += derivatives[-1][neuron_first_layer]['s'] * weights[1][neuron_first_layer][input_value_index]
        first_layer_der.append(y_derivative)

    return derivatives, first_layer_der


def arctan_derivative(x):
    return 1 / (1 + x**2)


def train():
    pass


if __name__ == '__main__':
    print("Hello World :)")
