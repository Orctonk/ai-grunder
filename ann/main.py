import numpy as np
from ann import ANN
from util import *
from layer import Layer

def main():
    (train_images,train_labels,test_images,test_labels) = load_data()
    train_labels_ohv = labels_to_1_hot(train_labels)

    network = ANN(784, 'mean_square_error')
    network.add_layer(3)
    network.add_layer(3)
    network.add_layer(10, 'tanh')

    test_out = network.eval(test_images)

    test_accuracy = calculate_accuracy(test_labels,test_out.argmax(1))
    print("Test accuracy: {}%".format(test_accuracy * 100))

    network.train(train_images,train_labels_ohv,100)

    test_out = network.eval(test_images)

    test_accuracy = calculate_accuracy(test_labels,test_out.argmax(1))
    print("Test accuracy: {}%".format(test_accuracy * 100))
    

if __name__ == "__main__":
    main()
