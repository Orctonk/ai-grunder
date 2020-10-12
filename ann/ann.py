import numpy as np
import time
from layer import Layer
from loss import *

class ANN():
    """ Base class for Artificial Neural Network.

    This class is used to create and train neural network as well as evaluate input.

    Sample usage: ::

     >>> network = ANN(100) 
     >>> network.add_layer(10) 
     >>> network.add_layer(10) 
     >>> network.train(train_data,train_labels,200)
     >>> res = network.eval(test_data)

    This creates a new network with 100 inputs one hidden layer with ten neurons
    and an output layer with 10 outputs.

    See add_layer, eval and train for more information
    
    """
    def __init__(self, inputsize, loss = 'mean_sqaure_error', loss_deriv = None, regularization=None, regularization_deriv=None, lambd=0):
        """ Creates a new ANN object with only input layer

        Parameters
        ----------
        inputsize: The size of the input layer
        loss: The loss function to be used when training the network
        loss_deriv: The derivative of the loss function with respect to each output, 
            is only needed if custom loss is used
        regularization: The regularization method used when training the network, optional.
        regularization_deriv: The derivative of the regularization function used, 
            only needed if custom regularization is used.

        lamd: The lambda used in regularization.
        """
        self.layers = []
        self.lastsize = inputsize
        self.inputsize = inputsize
        self.lambd = lambd

        # Set functions based on parameters
        if loss == 'mean_square_error':
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

    def add_layer(self, size, activation='leaky_relu', activation_deriv=None):
        """Adds a single layer to the network at the end

        Parameters provided are provided to Layer to create layer. See Layer class
        for more information

        Parameters
        ----------
        size: The amount of neurons in the layer.
        activation: The activation function used by this layer, 
            valid predefined strings are 'leaky_relu' and 'tanh'.

        activation_deriv: The derivative of the activation function with respect to each logit.
        """
        self.layers.append(Layer(size,self.lastsize,activation,activation_deriv))
        self.lastsize = size

    def eval(self, inputarr):
        """Passes the input through the network producing output
        
        Parameters
        ----------
        inputarr: The input to pass through the network.
        
        Return
        ------
        The output of the network
        """
        out = inputarr
        for l in self.layers:
            out = l.eval(out)

        return out 

    def _backpropagate(self, cost_deriv,loss,learning_rate):
        """Performs backpropagation and updates the network

        Parameters
        ----------
        cost_deriv: The derivative of the cost excluding regularization with respect to the outputs.
        loss: The calculated loss of the output.
        learning_rate: The current learning rate of the network.
        """
        last_deriv = cost_deriv
        for l in self.layers[-1::-1]:
            (last_deriv,_,_) = l.backpropagate(last_deriv,loss,learning_rate)
            l.weights -= learning_rate * self.regularization_deriv(l.weights,self.lambd)

    def train(self,data,labels,epochs,batch_size=100, learing_rate=0.1, decay=0, print_progress=False, test_data=None, test_labels=None, test_accuracy=None):
        """Trains the network using backpropagation and Gradient Descent

        Parameters
        ----------
        data: The training data
        labels: The training labels.
        epochs: The amount of iterations for which to run training.
        batch_size: The size of each batch of training data.
        learning_rate: A factor with which the gradient is multiplied before
            network parameters are updated.
        decay: The rate at which the learning rate decays with each epoch.
        print_progress: Whether or not to print training progress.
        test_data: Data to test the network on as it is trained.
        test_labels: Labels to compare test output with.
        test_accuracy: Function which calculates the accuracy of the test classification

        Return
        ------
        A 3-tuple containing the training loss, test loss and test accuracy for each epoch.
        """
        start = time.perf_counter()
        batch_count = np.math.ceil(data.shape[0]/batch_size)
        train_loss_history = []
        test_loss_history = []
        test_accuracy_history = []
        train_accuracy_history = []
        for e in range(1,epochs+1):
            epoch_loss = 0
            for i in range(batch_count):
                # Collect weights for regularization
                weights = np.array([])
                for l in self.layers:
                    weights = np.append(weights,l.weights)

                # Split into batch
                batch_data = data[i*batch_size:(i+1)*batch_size]
                batch_labels = labels[i*batch_size:(i+1)*batch_size]

                # Calculate loss
                output = self.eval(batch_data)
                reg = self.regularization(weights,self.lambd)
                loss = self.loss(output,batch_labels) + reg
                cost_deriv = self.loss_deriv(output,batch_labels)

                # Backpropagate for batch
                self._backpropagate(cost_deriv,loss,learing_rate)

                epoch_loss += loss.sum() / loss.size
            # Decay learning rate
            learing_rate = learing_rate * (1/(1 + e*decay))

            # Record loss
            train_loss_history.append(epoch_loss/batch_count)
            if print_progress:
                elapsed = time.perf_counter() - start
                print(f"\rElapsed time {elapsed:.1f}s, Epoch {e}: Loss = {epoch_loss/batch_count:.6f}",end="")
            if not test_data is None and not test_labels is None:
                test_output = self.eval(test_data)
                test_loss = self.loss(test_output,test_labels) + self.regularization(weights,self.lambd)
                test_loss_history.append(test_loss.sum()/test_loss.size)
                if test_accuracy != None:
                    train_output = self.eval(data)
                    test_accuracy_history.append(test_accuracy(test_output,test_labels))
                    train_accuracy_history.append(test_accuracy(train_output,labels))
        if print_progress:
            print("")
        return (np.array(train_loss_history),np.array(test_loss_history), np.array(train_accuracy_history), np.array(test_accuracy_history))


