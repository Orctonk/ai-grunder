import numpy as np
from layer import Layer
from loss import *

class ANN():
    def __init__(self, inputsize, loss = 'categorical_crossentropy', loss_deriv = None):
        self.layers = []
        self.lastsize = inputsize
        self.inputsize = inputsize
        if loss == 'categorical_crossentropy':
            self.loss = categorical_crossentropy
            self.loss_deriv = cc_deriv
        elif loss == 'mean_square_error':
            self.loss = mean_square_error
            self.loss_deriv = mse_deriv
        else:
            self.loss = loss
            self.loss_deriv = loss_deriv

    def add_layer(self, size, activation='relu', activation_deriv=None):
        self.layers.append(Layer(size,self.lastsize,activation,activation_deriv))
        self.lastsize = size

    def eval(self, inputarr):
        out = inputarr
        for l in self.layers:
            out = l.eval(out)

        return out 

    def backpropagate(self, cost_deriv,loss):
        last_deriv = cost_deriv
        for l in self.layers[-1::-1]:
            (last_deriv,_,_) = l.backpropagate(last_deriv,loss)

    def test(self,data,labels):
        output = self.eval(data)
        return self.loss(output,labels)

    def train(self,data,labels,epochs):
        for e in range(1,epochs+1):
            output = self.eval(data)
            loss = self.loss(output,labels)
            loss_per_data = loss.sum(1)
            epoch_loss = loss_per_data.sum() / loss_per_data.size
            print("Epoch {}: Loss = {}".format(e,epoch_loss))
            cost_deriv = self.loss_deriv(output,labels)
            self.backpropagate(cost_deriv,loss_per_data)

        pass


