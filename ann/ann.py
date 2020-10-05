import numpy as np
import time
from layer import Layer
from loss import *

class ANN():
    def __init__(self, inputsize, loss = 'categorical_crossentropy', loss_deriv = None, regularization=None, regularization_deriv=None, lambd=0.001):
        self.layers = []
        self.lastsize = inputsize
        self.inputsize = inputsize
        self.lambd = lambd
        if loss == 'categorical_crossentropy':
            self.loss = categorical_crossentropy
            self.loss_deriv = cc_deriv
        elif loss == 'mean_square_error':
            self.loss = mean_square_error
            self.loss_deriv = mse_deriv
        else:
            self.loss = loss
            self.loss_deriv = loss_deriv

        if regularization == 'L2':
            self.regularization = l2_regularization
            self.regularization_deriv = l2_deriv
        elif regularization == None:
            self.regularization = lambda w,l: 0
            self.regularization_deriv = lambda w,l: np.zeros_like(w)
        else:
            self.regularization = regularization
            self.regularization_deriv = regularization_deriv

    def add_layer(self, size, activation='relu', activation_deriv=None):
        self.layers.append(Layer(size,self.lastsize,activation,activation_deriv))
        self.lastsize = size

    def eval(self, inputarr):
        out = inputarr
        for l in self.layers:
            out = l.eval(out)

        return out 

    def backpropagate(self, cost_deriv,loss,learning_rate):
        last_deriv = cost_deriv
        for l in self.layers[-1::-1]:
            (last_deriv,_,_) = l.backpropagate(last_deriv,loss,learning_rate)
            l.weights -= learning_rate * self.regularization_deriv(l.weights,self.lambd)

    def train(self,data,labels,epochs,batch_size=100, learing_rate=0.2):
        start = time.perf_counter()
        batch_count = np.math.ceil(data.shape[0]/batch_size)
        loss_history = []
        for e in range(1,epochs+1):
            epoch_loss = 0
            for i in range(batch_count):
                weights = np.array([])
                for l in self.layers:
                    weights = np.append(weights,l.weights)
                batch_data = data[i*batch_size:(i+1)*batch_size]
                batch_labels = labels[i*batch_size:(i+1)*batch_size]
                output = self.eval(batch_data)
                reg = self.regularization(weights,self.lambd)
                loss = self.loss(output,batch_labels) + reg
            
                cost_deriv = self.loss_deriv(output,batch_labels)
                self.backpropagate(cost_deriv,loss,learing_rate)
                epoch_loss += loss.sum() / loss.size
            elapsed = time.perf_counter() - start
            loss_history.append(epoch_loss/batch_count)
            print(f"\rElapsed time {elapsed:.1f}s, Epoch {e}: Loss = {epoch_loss/batch_count:.6f}",end="")
        print("")
        return np.array(loss_history)


