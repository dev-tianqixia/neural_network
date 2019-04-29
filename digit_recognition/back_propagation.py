import random as rd

import numpy as np
from matplotlib import pyplot as plt

import parser

def sigmoid(z, derivative=False):
    sig_res = 1. / (1. + np.exp(-z))
    if(derivative):
        return sig_res * (1. - sig_res)
    else:
        return sig_res

class NeuralNetwork:
    """
    a simple neural net for hand written digit recognition.
    """
    def __init__(self, dimensions):
        # dimensions of the neural network: e.g. [784, 16, 10]
        self.dimensions = dimensions
        # randomly initialize weights and biases
        # i: number of neurons in layer l; j: number of neurons in layer l+1
        self.weights = [np.random.randn(j, i) for i, j in zip(dimensions[:-1], dimensions[1:])]
        # each activation has a bias associated with it (except input layer)
        self.biases = [np.random.randn(i, 1) for i in dimensions[1:]]
        # len(self.weights) == len(self.biases) == len(self.dimensions) - 1


    def train(self, inputs, epochs, batch_size, learning_rate):
        """
        divide inputs into multiple batches and apply stochastic gradient decent
        Args:
            inputs: a list of tuples (pixel_data, expected_digit)
        Returns:
            None
        """
        # repeat the algorithm "epochs" number of times
        for _ in range(epochs):
            # shuffle the inputs
            rd.shuffle(inputs)
            training_data, validation_data = inputs[:50000], inputs[50000:]
            # split inputs into batches, each batch contain batch_size training data
            batches = [training_data[start:start + batch_size] for start in range(0, len(training_data), batch_size)]
            for batch in batches:
                self.process_batch(batch, learning_rate)
            self.test(validation_data)


    def process_batch(self, batch, learning_rate):
        accum_delta_weight = [np.zeros(weight.shape) for weight in self.weights]
        accum_delta_bias = [np.zeros(bias.shape) for bias in self.biases]

        for pixel_data, expected_digit in batch:
            delta_weight, delta_bias = self.back_propagation(pixel_data, expected_digit)
            accum_delta_weight = [accum_w + delta_w for accum_w, delta_w in zip(accum_delta_weight, delta_weight)]
            accum_delta_bias = [accum_b + delta_b for accum_b, delta_b in zip(accum_delta_bias, delta_bias)]
        
        self.weights = [current_w - learning_rate * accum_w / len(batch) for current_w, accum_w in zip(self.weights, accum_delta_weight)]
        self.biases = [current_b - learning_rate * accum_b / len(batch) for current_b, accum_b in zip(self.biases, accum_delta_bias)]


    def back_propagation(self, pixel_data, expected_digit):
        delta_weight = [np.zeros(weight.shape) for weight in self.weights]
        delta_bias = [np.zeros(bias.shape) for bias in self.biases]
        # forward propagation to compute weighted sum(z) and activation(a) in each layer
        activation = pixel_data # original input
        activations, weighted_sums = [activation], []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            weighted_sums.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # calculate error for the output layer
        error = (activations[-1] - expected_digit) * sigmoid(weighted_sums[-1], derivative=True)
        delta_weight[-1] = np.dot(error, activations[-2].transpose())
        delta_bias[-1] = error
        # calculate error for remaining hidden layers
        for i in range(2, len(self.dimensions)):
            error = np.dot(self.weights[-i+1].transpose(), error) * sigmoid(weighted_sums[-i], derivative=True)
            delta_weight[-i] = np.dot(error, activations[-i-1].transpose())
            delta_bias[-i] = error
        
        return (delta_weight, delta_bias)
    
    
    def test(self, inputs):
        correct_count = 0
        for pixel_data, expected_digit in inputs:
            activation = pixel_data
            for w, b in zip(self.weights, self.biases):
                activation = sigmoid(np.dot(w, activation) + b)
            if np.argmax(activation) == np.argmax(expected_digit):
                correct_count += 1
        print("{0} / {1}".format(correct_count, len(inputs)))


if __name__ == "__main__":
    inputs = parser.Parser().parse('./data/train-images-idx3-ubyte', './data/train-labels-idx1-ubyte')
    net = NeuralNetwork([784, 64, 10])
    net.train(inputs, 5, 10, 1.)
