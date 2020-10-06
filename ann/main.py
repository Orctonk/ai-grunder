import numpy as np
import matplotlib.pyplot as plt
from ann import ANN
from util import *
from layer import Layer

def main():
    (train_images,train_labels,test_images,test_labels) = load_data()
    train_labels_ohv = labels_to_1_hot(train_labels)

    network = ANN(784, 'mean_square_error',regularization=None,lambd=0.005)
    network.add_layer(10,'leaky_relu')
    network.add_layer(10,'leaky_relu')
    network.add_layer(10, 'softmax')

    test_out = network.eval(test_images)

    test_accuracy = calculate_accuracy(test_labels,test_out.argmax(1))
    print("Test accuracy: {}%".format(test_accuracy * 100))

    loss = network.train(train_images,train_labels_ohv,1000,100,0.01)
    xaxis = [x for x in range(1000)]

    plt.plot(xaxis,loss)
    plt.show()

    train_out = network.eval(train_images)
    test_out = network.eval(test_images)

    train_accuracy = calculate_accuracy(train_labels,train_out.argmax(1))
    print("Train accuracy: {}%".format(train_accuracy * 100))

    test_accuracy = calculate_accuracy(test_labels,test_out.argmax(1))
    print("Test accuracy: {}%".format(test_accuracy * 100))
    

if __name__ == "__main__":
    main()
