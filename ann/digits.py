import numpy as np
import matplotlib.pyplot as plt
import sys
from ann import ANN
from util import *
from layer import Layer

devel = False

def main():
    if len(sys.argv) != 4:
        print("Invalid number of arguments.")
        print(f"Expected usage: python {sys.argv[0]} train_images train_labels validation_images")
        return
    
    test_label_path = None
    history_images = None
    history_labels = None
    history_accuracy = None
    if devel:
        test_label_path = "validation-labels.txt"

    (train_images,train_labels,test_images,test_labels) = load_data(sys.argv[1],sys.argv[2],sys.argv[3],test_label_path)
    train_labels_ohv = labels_to_1_hot(train_labels)

    if devel:
        test_labels_ohv = labels_to_1_hot(test_labels)
        history_images = test_images
        history_labels = test_labels_ohv
        history_accuracy = calculate_accuracy

    network = ANN(784, 'mean_square_error',regularization='L2',lambd=0.01)
    network.add_layer(10,'leaky_relu')
    network.add_layer(10,'leaky_relu')
    network.add_layer(10, 'tanh')

    (train_loss,test_loss,train_acc,test_acc) = network.train(train_images,train_labels_ohv,40,10,0.1,0.1/40,devel,history_images,history_labels,history_accuracy)

    test_out = network.eval(test_images)

    if not devel:
        labels = test_out.argmax(1)
        for i in labels:
            print(f"{i}")
    else:
        xaxis = [x for x in range(train_loss.size)]
        plt.figure(1)
        plt.plot(xaxis,train_loss,xaxis,test_loss)
        plt.legend(("Training loss", "Test loss"))
        plt.figure(2)
        line = plt.plot(xaxis,train_acc,xaxis,test_acc)
        plt.legend(("Training accuracy", "Test accuracy"))
        plt.show()

if __name__ == "__main__":
    main()
