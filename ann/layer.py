import numpy as np
from activation import *

class Layer():
    """Internal representation of a single layer in an artificial neural network
    """
    def __init__(self,size,prevsize, activation = 'leaky_relu', activation_deriv = None):
        """Creates a new layer

        Parameters
        ----------
        size: The amount of neurons in the network.
        prevsize: The amount of neurons in the previous layer.
        activation: The activation function used by this layer, can be strings 
            'tanh', 'leaky_relu' or a function
        activation_deriv: The derivative of the activation function if a custom is used.
            Should return the Jacobian.
        """
        self.size = size
        self.prevsize = prevsize
        
        self.biases = np.zeros(size)

        # Initializing weights randomly based on https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78
        if activation == 'leaky_relu':
            self.activation = leaky_relu
            self.activation_deriv = leaky_relu_deriv
            self.weights = np.random.randn(prevsize,size) * np.sqrt(2/prevsize)
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv
            self.weights = np.random.randn(prevsize,size) * np.sqrt(1/prevsize)
        else:
            self.activation = activation
            self.activation_deriv = activation_deriv
            self.weights = np.random.randn(prevsize,size) * np.sqrt(2/(prevsize+size))

    def eval(self, inputarr):
        """Calculates the activations of the layer based on activation of precious layer

        Parameters
        ----------
        inputarr: The activations of the previous layer

        Return
        ------
        The activation of this layer
        """
        self.last_activation = inputarr.reshape(-1,1,self.prevsize)
        self.last_raw = (np.dot(self.last_activation,(self.weights)) + self.biases).reshape(-1,self.size)
        return self.activation(self.last_raw)

    def backpropagate(self, last_derivs, loss, learning_rate):
        """Propagates an error back through this layer and updates parameters based on gradient

        Parameters
        ----------
        last_derivs: The derivatives of the cost with respect to this layer's activation.
        loss: The total loss of the last activation
        learning_rate: The learning rate parameter of the ANN

        Return
        ------
        The derivative with respect to the previous layer's activations
        """
        last_derivs = last_derivs.reshape(-1,1,self.size)
        
        bias_derivs = np.matmul(last_derivs, self.activation_deriv(self.last_raw))
        weight_derivs = np.matmul(self.last_activation.transpose(0,2,1),bias_derivs)
        last_activation_derivs = np.matmul(bias_derivs, self.weights.transpose())

        self.weights -= np.average((loss*learning_rate)[:,None,None] * weight_derivs, 0)
        self.biases -= np.average((loss*learning_rate)[:,None,None] * bias_derivs, 0)[0]

        return (last_activation_derivs,weight_derivs,bias_derivs)

