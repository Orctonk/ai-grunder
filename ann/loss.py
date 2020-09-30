import numpy as np

def categorical_crossentropy(output,target):
    return -np.multiply(target,np.log(output))

def cc_deriv(output,target):
    return -np.divide(target,output)

def mean_square_error(output,target):
    return np.power(np.subtract(output,target),2)

def mse_deriv(output,target):
    return -2*target + 2*output