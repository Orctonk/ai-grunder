import numpy as np
from activation import *

class Layer():
    def __init__(self,size,prevsize, activation = 'relu', activation_deriv = None):
        self.size = size
        self.prevsize = prevsize
        
        self.biases = np.zeros(size)

        # Initializing weights randomly based on https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78
        if activation == 'relu':
            self.activation = relu
            self.activation_deriv = relu_deriv
            self.weights = np.random.randn(prevsize,size) * np.sqrt(2/prevsize)
        elif activation == 'leaky_relu':
            self.activation = leaky_relu
            self.activation_deriv = leaky_relu_deriv
            self.weights = np.random.randn(prevsize,size) * np.sqrt(2/prevsize)
        elif activation == 'softmax':
            self.activation = softmax
            self.activation_deriv = softmax_deriv
            self.weights = np.random.randn(prevsize,size) * np.sqrt(2/(prevsize+size))
        elif activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_deriv = sigmoid_deriv
            self.weights = np.random.randn(prevsize,size) * np.sqrt(2/(prevsize+size))
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv
            self.weights = np.random.randn(prevsize,size) * np.sqrt(1/prevsize)
        else:
            self.activation = activation
            self.activation_deriv = activation_deriv
            self.weights = np.random.randn(prevsize,size) * np.sqrt(2/(prevsize+size))

    def eval(self, inputarr):
        self.last_activation = inputarr.reshape(-1,1,self.prevsize)
        self.last_raw = (np.dot(self.last_activation,(self.weights)) + self.biases).reshape(-1,self.size)
        return self.activation(self.last_raw)

    def backpropagate(self, last_derivs, loss, learning_rate):
        last_derivs = last_derivs.reshape(-1,1,self.size)
        
        bias_derivs = np.matmul(last_derivs, self.activation_deriv(self.last_raw))
        weight_derivs = np.matmul(self.last_activation.transpose(0,2,1),bias_derivs)
        last_activation_derivs = np.matmul(bias_derivs, self.weights.transpose())

        self.weights -= np.average((loss*learning_rate)[:,None,None] * weight_derivs, 0)
        self.biases -= np.average((loss*learning_rate)[:,None,None] * bias_derivs, 0)[0]

        return (last_activation_derivs,weight_derivs,bias_derivs)

