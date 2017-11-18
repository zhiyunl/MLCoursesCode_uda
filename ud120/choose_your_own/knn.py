#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17/11/18 21:14
# @Author  : zhiyun
# @Site    : 
# @File    : knn.py
# @Software: PyCharm

import sys

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from choose_your_own.prep_terrain_data import makeTerrainData

sys.path.append("../tools/")

# features_train, features_test, labels_train, labels_test = preprocess()
features_train, labels_train, features_test, labels_test = makeTerrainData()

# max_acc = [clf, n_neighbors, weights,acc]
max_acc = [KNeighborsClassifier(), 0, 'distance', 0.00]


def classify(n_neighbors=5, weights='uniform'):
    clf = KNeighborsClassifier(n_neighbors, weights)

    # t = time()
    clf.fit(features_train, labels_train)
    # print('training time is ', round(time() - t, 3), 's')

    # t = time()
    pred = clf.predict(features_test)
    # print('predicting time is ', round(time() - t, 3), 's')

    # t = time()
    acc = accuracy_score(labels_test, pred)
    # print('accuracy time is ', round(time() - t, 3), 's')
    # another way, it combines prediction and accuracy calculation
    # acc = clf.score(features_test,labels_test)

    print('acc with neighbors ', n_neighbors, ',', weights, 'algorithm = ', round(acc, 3))
    if acc >= max_acc[3]:
        max_acc[0] = clf
        max_acc[1] = n_neighbors
        max_acc[2] = weights
        max_acc[3] = acc


def knn_main():
    for weights in ['uniform', 'distance']:
        for n_neighbors in range(100):
            n_neighbors = n_neighbors + 1
            classify(n_neighbors, weights)
    print('the best parameter is ', max_acc[1:3], '\nthe accuracy is ', round(max_acc[3], 4))
    return max_acc[0]
