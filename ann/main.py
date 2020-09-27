import numpy as np
from ann import ANN
from util import *
from layer import Layer

def main():
    (train_images,train_labels,test_images,test_labels) = load_data()
    train_images = train_images.astype(float) / 255
    train_labels = train_labels.astype(int)

    # Process labels to 1-hot form
    # see https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-1-hot-encoded-numpy-array
    train_labels_ohv = np.zeros((train_labels.size,train_labels.max()+1))
    train_labels_ohv[np.arange(train_labels.size),train_labels] = 1

    network = ANN(784)
    network.add_layer(3)
    network.add_layer(3)
    network.add_layer(10,'softmax')

    network.train(train_images,train_labels_ohv,100)

if __name__ == "__main__":
    main()
