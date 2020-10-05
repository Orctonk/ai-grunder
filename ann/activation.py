import numpy as np

def relu(inp):
    return np.maximum(inp,0)

def relu_deriv(inp):
    inp[inp<0] = 0
    inp[inp>=0] = 1
    n = inp.shape[1]
    out = np.zeros((inp.shape[0],n,n))
    out[:,np.arange(n),np.arange(n)] = inp
    return out

def leaky_relu(inp):
    out = np.zeros_like(inp)
    out[inp>=0] = inp[inp>=0]
    out[inp<0] = inp[inp<0]
    return out

def leaky_relu_deriv(inp):
    inp[inp<0] = 0.01
    inp[inp>=0] = 1
    n = inp.shape[1]
    out = np.zeros((inp.shape[0],n,n))
    out[:,np.arange(n),np.arange(n)] = inp
    return out

def sigmoid(inp):
    return 1 / (1 + np.exp(-inp)) 

def sigmoid_deriv(inp):
    f = sigmoid(inp)
    n = inp.shape[1]
    out = np.zeros((inp.shape[0],n,n))
    out[:,np.arange(n),np.arange(n)] = np.multiply(f,(1 - f))
    return out

def softmax(inp):
    expterm = np.exp(inp)
    return expterm / np.sum(expterm,1)[:,None] # https://stackoverflow.com/questions/16202348/numpy-divide-row-by-row-sum

def softmax_deriv(inp):
    inp3d = np.atleast_3d(inp)
    inp3d = inp3d.reshape(inp3d.shape[0],1,inp.shape[-1])
    
    pass

def tanh(inp):
    return np.tanh(inp)

def tanh_deriv(inp):
    n = inp.shape[1]
    out = np.zeros((inp.shape[0],n,n))
    out[:,np.arange(n),np.arange(n)] = 1 - np.tanh(inp) ** 2
    return out