import numpy as np
from layer import Layer
from loss import *

class ANN():
    def __init__(self, inputsize):
        self.layers = []
        self.lastsize = inputsize
        self.inputsize = inputsize

    def add_layer(self, size, activation='relu'):
        self.layers.append(Layer(size,self.lastsize,activation))
        self.lastsize = size

    def eval(self, inputarr):
        out = inputarr
        for l in self.layers:
            out = l.eval(out)

        return out 

    def backpropagate(self, cost_deriv):
        last_deriv = cost_deriv
        for l in self.layers[-1::-1]:
            (last_deriv,_,_) = l.backpropagate(last_deriv)

    def test(self,data,labels):
        output = self.eval(data)
        return categorical_crossentropy(output,labels)

    def train(self,data,labels,epochs):
        for e in range(1,epochs+1):
            output = self.eval(data)
            loss = categorical_crossentropy(output,labels)
            loss_per_data = loss.sum(1)
            epoch_loss = loss_per_data.sum() / loss_per_data.size
            print("Epoch {}: Loss = {}".format(e,epoch_loss))
            cost_deriv = cc_deriv(output,labels)
            self.backpropagate(cost_deriv)

        pass


