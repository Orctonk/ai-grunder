import numpy as np

def relu(inp):
    return np.maximum(np.zeros(np.shape(inp)),inp)

def sigmoid(inp):
    return 1/(1+np.exp(inp))

def softmax(inp):
    expterm = np.exp(inp)
    return expterm / np.sum(expterm,1)[:,None] # https://stackoverflow.com/questions/16202348/numpy-divide-row-by-row-sum