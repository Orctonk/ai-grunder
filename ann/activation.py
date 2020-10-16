import numpy as np

def leaky_relu(inp):
    """Calcualtes a modified ReLU which fixes the problem of dying neurons

    Parameters
    ----------
    inp: The logits to perform the activation on

    Return
    ------
    The result of the activation function
    """
    out = np.zeros_like(inp)
    out[inp>=0] = inp[inp>=0]
    out[inp<0] = inp[inp<0] * 0.1
    return out

def leaky_relu_deriv(inp):
    """Calculates the derivative of the leaky ReLU activation function

    Parameters
    ----------
    inp: The logits used for the activation

    Return
    ------
    The Jacobian for the leaky ReLU functions for the last logits
    """
    inp[inp<0] = 0.1
    inp[inp>=0] = 1
    n = inp.shape[1]
    out = np.zeros((inp.shape[0],n,n))
    out[:,np.arange(n),np.arange(n)] = inp
    return out

def tanh(inp):
    """Calcualtes the elementwise hyperbolic tangents

    Parameters
    ----------
    inp: The logits to perform the activation on

    Return
    ------
    The result of the activation function
    """
    return np.tanh(inp)

def tanh_deriv(inp):
    """Calculates the derivative of the tanhactivation function

    Parameters
    ----------
    inp: The logits used for the activation

    Return
    ------
    The Jacobian for the tanh function for the last logits
    """
    n = inp.shape[1]
    out = np.zeros((inp.shape[0],n,n))
    out[:,np.arange(n),np.arange(n)] = 1 - np.tanh(inp) ** 2
    return out