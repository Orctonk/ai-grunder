import numpy as np

def categorical_crossentropy(output,target):
    return -np.multiply(target,np.log(output+1e-9)).sum(1)

def cc_deriv(output,target):
    return -np.divide(target,output)

def mean_square_error(output,target):
    return np.average(np.power(np.subtract(output,target),2),1)

def mse_deriv(output,target):
    return -2*target + 2*output

def l2_regularization(weights, lambd):
    return (lambd / 2) * np.power(weights,2).sum()

def l2_deriv(weights,lambd):
    return lambd * weights
