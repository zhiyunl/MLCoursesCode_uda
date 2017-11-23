#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17/11/20 22:57
# @Author  : zhiyun
# @Site    : 
# @File    : adaboost.py
# @Software: PyCharm

import sys

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

from choose_your_own.prep_terrain_data import makeTerrainData

sys.path.append("../tools/")

# features_train, features_test, labels_train, labels_test = preprocess()
features_train, labels_train, features_test, labels_test = makeTerrainData()

# max_acc = [clf, algorithm, learning_rate, n_estimators, acc]
max_acc = [AdaBoostClassifier(n_estimators=50), 'SAMME.R', 1.0, 0, 0.00]


def classify(n_estimators=100):
    clf = AdaBoostClassifier(n_estimators=n_estimators)
    # scores = cross_val_score(clf,features_train,labels_train)
    # print(scores.mean())
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    acc = accuracy_score(labels_test, pred)
    # acc = clf.score(features_test,labels_test)
    print('acc with estimators ', n_estimators, ' = ', acc)
    if acc >= max_acc[4]:
        max_acc[0] = clf
        max_acc[3] = n_estimators
        max_acc[4] = acc
        # return clf


def ada_main():
    for n_estimators in range(1, 50):
        classify(n_estimators)
    print('the best parameter is ', max_acc[1:4], '\nthe accuracy is ', round(max_acc[4], 4))
    return max_acc[0]
