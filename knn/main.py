from knn import *
from itertools import combinations
import matplotlib as mpl
import matplotlib.pyplot as plt

colors = ["b","g","r","c","m","y","k","w"]

def get_iris_scores(k_max,split):
    exclude = [0,1,2,4,5]

    features = [0,1,2,3]
    feature_combinations = combinations(features,2)

    train_scores = []
    test_scores = []

    combstr = []
    fnum = 0

    for (i,c) in enumerate(feature_combinations):
        if i in exclude:
            continue
        x_train,x_test,y_train,y_test = load_dataset("iris",c,split)
        combstr.append(str(c))
        train_scores.append([])
        test_scores.append([])

        for k in range(1,k_max+1):
            knn = train_knn(x_train,y_train,k)
            (train,test) = evaluate_knn(knn,x_train,y_train,x_test,y_test)
            train_scores[fnum].append(train)
            test_scores[fnum].append(test)
        fnum += 1

    return (train_scores,test_scores,combstr)

def get_cancer_scores(k_max,split):
    features = [x for x in range(30)]
    exclude = [1,4,5,8,9,10,11,12,14,15,16,17,18,19,21,24,25,26,28,29]

    train_scores = []
    test_scores = []

    combstr = []

    features = []
    for i in range(30):
        if not i in exclude:
            features.append(i)

    x_train,x_test,y_train,y_test = load_dataset("breastcancer",features,split)
    combstr.append(str(features))
    train_scores.append([])
    test_scores.append([])
    
    for k in range(1,k_max+1):
        knn = train_knn(x_train,y_train,k)
        (train,test) = evaluate_knn(knn,x_train,y_train,x_test,y_test)
        train_scores[0].append(train)
        test_scores[0].append(test)

    return train_scores,test_scores,combstr

def main():
    mpl.use("TkAgg")
    k_max = 30

    #(train_scores,test_scores,combstr) = get_iris_scores(k_max,0.66)
    (train_scores,test_scores,combstr) = get_cancer_scores(k_max,0.66)

    # Construct plot
    plt.title('KNN results')
    plt.xlabel('Number of Neighbors K')
    plt.ylabel('Accuracy')

    # Create x-axis
    xaxis = [x for x in range(1,k_max+1)]

    # Plot the test and training scores with labels
    print(len(train_scores))
    for i in range(len(train_scores)):
        plt.plot(xaxis, train_scores[i], colors[i]+"-", label='Training score ' + combstr[i])
        plt.plot(xaxis, test_scores[i], colors[i]+"--", label='Test score ' + combstr[i])

    # Show the figure
    plt.legend()
    plt.show()
    input()

if __name__ == "__main__":
    main()