import numpy as np
from activation import *

class Layer():
    def __init__(self,size,prevsize, activation = 'relu'):
        self.size = size
        self.prevsize = prevsize
        # Initializing weights randomly based on https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78
        # Is intended for ReLU activation though
        self.weights = np.random.randn(prevsize,size) * np.sqrt(2/prevsize)
        self.biases = np.zeros(size)

        if activation == 'relu':
            self.activation = relu
        elif activation == 'softmax':
            self.activation = softmax
        elif activation == 'sigmoid':
            self.activation = sigmoid
        else:
            self.activation = activation

    def eval(self, inputarr):
        raw = inputarr.dot(self.weights) + self.biases
        return self.activation(raw)
