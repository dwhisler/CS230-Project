import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from sklearn import svm, neural_network, naive_bayes, ensemble, neighbors, datasets
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score

from data_extraction import get_loan_data


def run_baseline_classifiers(data, target):
    #Includes a sample of several the machine learning classifiers
    classifiers = [
        # ensemble.RandomForestClassifier(random_state=0, n_estimators=10),
        # svm.SVC(gamma='auto'),
        # naive_bayes.GaussianNB(priors=None),
        # neighbors.KNeighborsClassifier(n_neighbors=5),
        neural_network.MLPClassifier(activation='relu', solver='adam', alpha=1e-5, hidden_layer_sizes=(100,30,), random_state=0, learning_rate_init=0.0001),
        ]
    features_train, features_test, labels_train, labels_test = train_test_split(data, target, test_size=0.1, random_state=5)


    for clf in classifiers:
        print(clf.__class__.__name__)
        #Parameters used in creating this classifier
        print('Parameters: ' + str(clf.get_params()))
        #Train & predict classifier
        clf.fit(features_train, labels_train)

        #Cross validation
        scores = cross_val_score(clf, features_train, labels_train, cv=5)
        print('Cross Validation Scores: ' + str(scores))
        print('Avg Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))

        #Test accuracy
        # train_accuracy = accuracy_score(labels_train, clf.predict(features_train))
        # print("Training accuracy: ", train_accuracy)
        test_accuracy = accuracy_score(labels_test, clf.predict(features_test))
        print("Test accuracy: ", test_accuracy)

if __name__ == '__main__':
    #X, y = datasets.load_iris(return_X_y=True)
    filename = "../data/accepted_2007_to_2018Q4.csv"
    cols_filename = "cols_all_test.txt"
    X, y = get_loan_data(filename, cols_filename, 10000)

    run_baseline_classifiers(X, y)
