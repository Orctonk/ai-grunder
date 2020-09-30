import numpy as np
from activation import *

class Layer():
    def __init__(self,size,prevsize, activation = 'relu', activation_deriv = None):
        self.size = size
        self.prevsize = prevsize
        # Initializing weights randomly based on https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78
        # Is intended for ReLU activation though
        self.weights = np.random.randn(prevsize,size) * np.sqrt(2/prevsize)
        self.biases = np.zeros(size)

        if activation == 'relu':
            self.activation = relu
            self.activation_deriv = relu_deriv
        elif activation == 'softmax':
            self.activation = softmax
            self.activation_deriv = softmax_deriv
        elif activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_deriv = sigmoid_deriv
        else:
            self.activation = activation
            self.activation_deriv = activation_deriv

    def eval(self, inputarr):
        self.last_activation = inputarr
        self.last_raw = inputarr.dot(self.weights) + self.biases
        return self.activation(self.last_raw)

    def backpropagate(self, last_derivs):
        bias_derivs = last_derivs.dot(self.activation_deriv(self.last_raw))
        weight_derivs = bias_derivs.dot(self.last_activation.transpose())
        last_activation_derivs = bias_derivs.dot(self.weights.transpose())

        return (last_activation_derivs,weight_derivs,bias_derivs)

