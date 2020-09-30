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
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv
        else:
            self.activation = activation
            self.activation_deriv = activation_deriv

    def eval(self, inputarr):
        self.last_activation = np.atleast_3d(inputarr)
        self.last_activation = self.last_activation.reshape(self.last_activation.shape[0],1,self.prevsize)
        self.last_raw = np.matmul(inputarr,(self.weights)) + self.biases
        return self.activation(self.last_raw)

    def backpropagate(self, last_derivs, loss):
        last_derivs = np.atleast_3d(last_derivs)
        last_derivs = last_derivs.reshape(last_derivs.shape[0],1,self.size)
        
        bias_derivs = np.matmul(last_derivs, self.activation_deriv(self.last_raw))
        weight_derivs = np.matmul(self.last_activation.transpose(0,2,1),bias_derivs)
        last_activation_derivs = np.matmul(bias_derivs, self.weights.transpose())
        
        self.weights += np.average((loss)[:,None,None] * weight_derivs, 0)
        self.biases += np.average((loss)[:,None,None] * bias_derivs, 0)[0]

        return (last_activation_derivs,weight_derivs,bias_derivs)

