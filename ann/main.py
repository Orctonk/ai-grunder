import numpy as np
from ann import *

def print_image(image):
    print("3"*100)
    print(len(image))
    for i in range(28):
        for x in image[i*28:i*28+28]:
            if int(x) > 10:
                print("#",end='')
            else:
                print(" ",end='')
        print("")


def main():
    (train_images,train_labels,test_images,test_labels) = load_data()
    print_image(test_images[0])
    print(test_labels[0])

if __name__ == "__main__":
    main()
