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

    traini = traini.astype(float) / 255
    testi = testi.astype(float) / 255
    trainl = trainl.astype(int)
    testl = testl.astype(int)

    return (traini,trainl,testi,testl)

def labels_to_1_hot(labels):
    # Process labels to 1-hot form
    # see https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-1-hot-encoded-numpy-array
    labels_ohv = np.zeros((labels.size,labels.max()+1))
    labels_ohv[np.arange(labels.size),labels] = 1
    return labels_ohv

def calculate_accuracy(target,classifications):
    accuracy_arr = np.zeros(target.shape)
    accuracy_arr[target==classifications] = 1
    return accuracy_arr.sum() / accuracy_arr.size