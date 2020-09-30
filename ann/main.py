import numpy as np
from ann import ANN
from util import *
from layer import Layer

def main():
    (train_images,train_labels,test_images,test_labels) = load_data()
    train_labels_ohv = labels_to_1_hot(train_labels)

    network = ANN(784)
    network.add_layer(3)
    network.add_layer(3)
    network.add_layer(10,'softmax')

    network.train(train_images,train_labels_ohv,100)

    test_out = network.eval(test_images)

    test_classifications = test_out.argmax(1)
    test_accuracy = np.multiply(test_labels,test_classifications).sum() / test_labels.size
    print("Test accuracy: {}%".format(test_accuracy))
    

if __name__ == "__main__":
    main()
