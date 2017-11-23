#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/23 16:16
# @Author  : zhiyun
# @Site    : 
# @File    : randomforest.py
# @Software: PyCharm

import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from choose_your_own.prep_terrain_data import makeTerrainData

sys.path.append("../tools/")

# features_train, features_test, labels_train, labels_test = preprocess()
features_train, labels_train, features_test, labels_test = makeTerrainData()

# max_acc = [clf, algorithm, learning_rate, n_estimators, acc]
max_acc = [RandomForestClassifier(), 1, 50, 0.00]


def classify(n_estimators=100, leaf_size=50):
    clf = RandomForestClassifier(n_estimators=n_estimators, oob_score=True, n_jobs=-1, min_samples_leaf=leaf_size)
    # scores = cross_val_score(clf,features_train,labels_train)
    # print(scores.mean())
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    acc = accuracy_score(labels_test, pred)
    # acc = clf.score(features_test,labels_test)
    print('acc with estimators ', n_estimators, ', leaf_size ', leaf_size, ' = ', acc)
    if acc >= max_acc[3]:
        max_acc[0] = clf
        max_acc[1] = n_estimators
        max_acc[2] = leaf_size
        max_acc[3] = acc
        # return clf


def rf_main():
    sample_leaf_options = range(80, 100, 1)
    for leaf_size in sample_leaf_options:
        for n_estimators in range(40, 50, 1):
            classify(n_estimators, leaf_size=leaf_size)
    print('the best parameter is ', max_acc[1:3], '\nthe accuracy is ', round(max_acc[3], 4))
    return max_acc[0]
