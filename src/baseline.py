import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#plt.use('PS')
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics, cross_validation
from sklearn.metrics import roc_curve
from sklearn.metrics import auc







import sklearn
from sklearn import svm, neural_network, naive_bayes, ensemble, neighbors, datasets
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score

from data_extraction import get_loan_data


def run_baseline_classifiers(data, target):
    #Includes a sample of several the machine learning classifiers
    classifiers = [
                   #linear_model.LogisticRegression(),
                   #ensemble.RandomForestClassifier(random_state=0, n_estimators=10),
                   #svm.SVC(gamma='auto'),
                   #naive_bayes.GaussianNB(priors=None),
                   #neighbors.KNeighborsClassifier(n_neighbors=5),
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
        
        
        # Plot count
        plt.figure()
        kf = KFold(n_splits=5, shuffle=True)
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0,1,100)
        i = 1
        for train_index, test_index in kf.split(features_train, labels_train):
            X_train, X_test = features_train.iloc[train_index,], features_train.iloc[test_index,]
            Y_train, Y_test = labels_train.iloc[train_index], labels_train.iloc[test_index]
            prediction = clf.fit(X_train,Y_train).predict_proba(X_test)
            fpr, tpr, t = roc_curve(Y_test, prediction[:, 1])
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i,
                                                                                 roc_auc))
            i= i+1
        plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, color='blue',
                 label=r'Mean ROC (AUC = %0.2f )' % (mean_auc),lw=2, alpha=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC for %s ' % (clf.__class__.__name__))
        plt.legend(loc="lower right")

        #Test accuracy
        train_accuracy = accuracy_score(labels_train, clf.predict(features_train))
        print("Training accuracy: ", train_accuracy)
        test_accuracy = accuracy_score(labels_test, clf.predict(features_test))
        
        
        print("Test accuracy: ", test_accuracy)

if __name__ == '__main__':
    #X, y = datasets.load_iris(return_X_y=True)
    filename = "../data/accepted_2007_to_2018Q4.csv"
    cols_filename = "cols_all_test.txt"
    X, y = get_loan_data(filename, cols_filename, 50000)

    run_baseline_classifiers(X, y)
    plt.show()
