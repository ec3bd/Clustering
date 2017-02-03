#Eamon Collins  ec3bd
#!/usr/bin/env python

import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import gzip
import numpy as np
import math
import sklearn.svm as sv
import sklearn.neighbors as nn
import sklearn.decomposition

def decision_tree(train, ytrain, test, ytest):
    dt = DecisionTreeClassifier(criterion='entropy', max_depth=15, min_samples_split=2)
    params = {'criterion':('gini', 'entropy'), 'max_depth':[3, 15, 20], 'min_samples_split':[2,5,10]}
    clf = GridSearchCV(dt, params)
    dt.fit(train, ytrain)
    clf.fit(train, ytrain)

    #print(clf.cv_results_['params'])
    #print(clf.cv_results_['mean_test_score'])
    y = dt.predict(test, ytest)

    #Your code here
    return y

def knn(train, ytrain, test, ytest):
    y = []
    clf = nn.KNeighborsClassifier(n_neighbors=3).fit(train, ytrain)
    y = clf.predict(test, ytest)
    #print(clf.score(train, ytrain))
    #Your code here
    return y

def svm(train, ytrain, test, ytest):
    y = []
    clf = sv.SVC(kernel = 'poly', C=15)
    clf.fit(train, ytrain)
    y = clf.predict(test, ytest)
    #Your code here
    return y

def pca_knn(train, ytrain, test, ytest):
    y = []
    pca = sklearn.decomposition.PCA(n_components=40, svd_solver='randomized')
    newtrain = pca.fit_transform(train)
    print(sum(pca.explained_variance_ratio_))
    newtest = pca.transform(test)
    clf = nn.KNeighborsClassifier(n_neighbors=3).fit(newtrain, ytrain)
    y = clf.predict(newtest, ytest)
    #Your code here
    return y

def pca_svm(train, ytrain, test, ytest):
    y = []
    pca = sklearn.decomposition.PCA(n_components=40, svd_solver='randomized')
    newtrain = pca.fit_transform(train)
    print(sum(pca.explained_variance_ratio_))
    newtest = pca.transform(test)
    clf = sv.SVC(kernel = 'poly', C=15)
    clf.fit(newtrain, ytrain)
    Accuracy = clf.score(newtest, ytest)
    y = clf.predict(newtest, ytest)
    #Your code here
    return y

if __name__ == '__main__':
    model = sys.argv[1]
    train = sys.argv[2]
    test = sys.argv[3]
    x = []
    t = []
    ytest = []
    ytrain = []
    with gzip.open(train, 'rb') as f:
        for line in f:
            sample = []
            line = line.split()
            for i in range(1,len(line) - 1):
                sample.append(float(line[i].decode("utf-8")))
            ytrain.append(int(math.floor(float(line[0].decode("utf-8")))))
            x.append(sample)
    with gzip.open(test, 'rb') as f:
        for line in f:
            sample = []
            line = line.split()
            for i in range(1,len(line) - 1):
                sample.append(float(line[i].decode("utf-8")))
            ytest.append(int(math.floor(float(line[0].decode("utf-8")))))
            t.append(sample)
    train = np.asmatrix(x)
    test = np.asmatrix(t)

    if model == "dtree":
        print(decision_tree(train, ytrain, test, ytest))
    elif model == "knn":
        print(knn(train, ytrain, test, ytest))
    elif model == "svm":
        print(svm(train, ytrain, test, ytest))
    elif model == "pcaknn":
        print(pca_knn(train, ytrain, test, ytest))
    elif model == "pcasvm":
        print(pca_svm(train, ytrain, test, ytest))
    else:
        print("Invalid method selected!")
