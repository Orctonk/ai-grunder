import numpy as np

def categorical_crossentropy(output,target):
    return -np.multiply(target,np.log(output))

def cc_deriv(output,target):
    return -np.divide(target,output)