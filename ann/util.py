import numpy as np

def read_labels(path):
    """Reads the label file provided
    
    Parameters
    ----------
    path: The path to the label file

    Return
    ------
    The labels read from the file
    """
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
    """Reads the image file provided
    
    Parameters
    ----------
    path: The path to the image file

    Return
    ------
    The image read from the file
    """
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

def load_data(traini_path,trainl_path,testi_path,testl_path=None):
    """Loads data from files

    Parameters
    ----------
    traini_path: The path to the training images.
    trainl_path: The path to the training labels.
    testi_path: The path to the test images.
    testl_path: The path to the test labels.

    Return
    ------
    A tuple with the loaded data in the same order as the arguments
    """
    traini = read_images(traini_path)
    testi = read_images(testi_path)

    trainl = read_labels(trainl_path)
    testl = None
    if testl_path != None:
        testl = read_labels("data/validation-labels.txt")
        testl = testl.astype(int)

    traini = traini.astype(float) / 255
    testi = testi.astype(float) / 255
    trainl = trainl.astype(int)

    return (traini,trainl,testi,testl)

def labels_to_1_hot(labels):
    """Processes labels into a more ANN-friendly format

    Parameters
    ----------
    labels: The labels to process

    Return
    ------
    An array where each label is substituted for a vector with a '1' on the position representing the label
    """
    # Process labels to 1-hot form
    # see https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-1-hot-encoded-numpy-array
    labels_ohv = np.zeros((labels.size,labels.max()+1))
    labels_ohv[np.arange(labels.size),labels] = 1
    return labels_ohv

def calculate_accuracy(target,classifications):
    """Calculates the accuracy of the classifications.

    Parameters
    ----------
    target: The target classifications.
    classification: The ANN classifications
    """
    accuracy_arr = np.zeros(target.shape)
    accuracy_arr[target==classifications] = 1
    return accuracy_arr.sum() / accuracy_arr.size