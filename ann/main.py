import numpy as np
import matplotlib.pyplot as plt
import sys
from ann import ANN
from util import *
from layer import Layer

def main():
    if len(sys.argv) != 4:
        print("Invalid number of arguments.")
        print(f"Expected usage: python {sys.argv[0]} train_images train_labels validation_images")
    (train_images,train_labels,test_images,test_labels) = load_data(sys.argv[1],sys.argv[2],sys.argv[3])
    train_labels_ohv = labels_to_1_hot(train_labels)

    network = ANN(784, 'mean_square_error',regularization='L2',lambd=0.005)
    network.add_layer(10,'leaky_relu')
    network.add_layer(10,'leaky_relu')
    network.add_layer(10, 'tanh')

    loss = network.train(train_images,train_labels_ohv,200,10,0.1,0.1/200,True)

    test_out = network.eval(test_images)

    labels = test_out.argmax(1)
    print(f"{labels.size} {4789}")
    for i in labels:
        print(f"{i}")
    

if __name__ == "__main__":
    main()
