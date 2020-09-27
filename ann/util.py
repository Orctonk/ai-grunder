import numpy as np

def read_labels(path):
    file = open(path,"r")

    reading_labels = False

    labels = []

    for l in file:
        if l.strip()[0] == '#':
            continue
        if reading_labels:
            labels.append(l[0])
        else:
            reading_labels = True
    
    file.close()

    return np.array(labels)

def read_images(path):
    file = open(path,"r")

    reading_images = False

    images = []

    for l in file:
        if l.strip()[0] == '#':
            continue
        if reading_images:
            images.append(l.split())
        else:
            reading_images = True

    file.close()

    return np.array(images)

def load_data():
    traini = read_images("data/training-images.txt")
    testi = read_images("data/validation-images.txt")

    trainl = read_labels("data/training-labels.txt")
    testl = read_labels("data/validation-labels.txt")

    return (traini,trainl,testi,testl)

