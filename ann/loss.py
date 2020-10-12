import numpy as np

def mean_square_error(output,target):
    """Calculates the mean square error loss of the output and target.

    Parameters
    ----------
    output: The output of the neural network.
    target: The expected output of the neural network.

    Return
    ------
    The mean square error loss of the output and the target
    """
    return np.average(np.power(np.subtract(output,target),2),1)

def mse_deriv(output,target):
    """Calculates the derivative of the mean square error loss with respect to each output activation

    Parameters
    ----------
    output: The output of the neural network.
    target: The expected output of the neural network.

    Return
    ------
    The derivative of the mean square error loss with respect to each output activation
    """
    return -2*target + 2*output

def l2_regularization(weights, lambd):
    """Calculates the regularization loss of all weights in the network and the provided lambda

    Parameters
    ----------
    weights: The weights used in the network.
    lambd: The regularization lambda used in the network

    Return
    ------
    The additional regularization loss of the weights
    """
    return (lambd / 2) * np.power(weights,2).sum()

def l2_deriv(weights,lambd):
    """Calculates the derivative of the regularization loss with respect to the given weights

    Parameters
    ----------
    weights: The weights to calculate the derivative for
    lambd: The lambda used in regularization

    Return
    ------
    The derivative of the regularization loss with respect to the weights
    """
    return lambd * weights
