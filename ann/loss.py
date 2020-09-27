import numpy as np

def categorical_crossentropy(output,target):
    return -np.multiply(target,np.log(output))