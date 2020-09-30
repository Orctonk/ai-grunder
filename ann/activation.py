import numpy as np

def relu(inp):
    return np.maximum(np.zeros(np.shape(inp)),inp)

def relu_deriv(inp):
    inp[inp<=0] = 0
    inp[inp>0]  = 1
    return inp

def sigmoid(inp):
    return 1/(1+np.exp(inp))

def sigmoid_deriv(inp):
    expterm = np.exp(inp)
    return expterm / (1 + expterm) ** 2

def softmax(inp):
    expterm = np.exp(inp)
    return expterm / np.sum(expterm,1)[:,None] # https://stackoverflow.com/questions/16202348/numpy-divide-row-by-row-sum

def softmax_deriv(inp):
    pass