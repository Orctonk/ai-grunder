from knn import *
from itertools import combinations
import matplotlib as mpl
import matplotlib.pyplot as plt

colors = ["b","g","r","c","m","y","k","w"]
exclude = [0,1,2]

def main():
    mpl.use("TkAgg")
    features = [0,1,2,3]
    k_max = 25
    train_scores = []
    test_scores = []
    
    feature_combinations = combinations(features,2)
    combstr = []
    fnum = 0
    for (i,c) in enumerate(feature_combinations):
        if i in exclude:
            continue
        x_train,x_test,y_train,y_test = load_dataset("iris",c,0.66)
        combstr.append(str(c))
        train_scores.append([])
        test_scores.append([])

        for k in range(1,k_max+1):
            knn = train_knn(x_train,y_train,k)
            (train,test) = evaluate_knn(knn,x_train,y_train,x_test,y_test)
            train_scores[fnum].append(train)
            test_scores[fnum].append(test)
        fnum += 1

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