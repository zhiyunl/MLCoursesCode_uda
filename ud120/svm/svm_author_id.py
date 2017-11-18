#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from tools.email_preprocess import preprocess
import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


# # shrink the data size
# features_train = features_train[:int(len(features_train) / 100)]
# labels_train = labels_train[:int(len(labels_train) / 100)]

# features_train, labels_train, features_test, labels_test = makeTerrainData()


#########################################################
### your code goes here ###
# return the name corresponding to the number
def check_name(pred):
    if pred == 1:
        return 'Chris'
    else:
        return 'Sara'


def svm_main():
    # for C = 1.0
    clf = SVC(kernel='rbf', C=10000)
    t = time()
    clf.fit(features_train, labels_train)
    print('training time is ', round(time() - t, 3), 's')

    t = time()
    pred = clf.predict(features_test)
    print('predicting time is ', round(time() - t, 3), 's')

    t = time()
    acc = accuracy_score(pred, labels_test)
    print('accuracy time is ', round(time() - t, 3), 's')
    # another way, it combines prediction and accuracy calculation
    # acc = clf.score(features_test,labels_test)

    print('acc = ', round(acc, 3))
    print('length of test is ', len(features_test))

    # mail number of chris
    cnt_array = np.bincount(pred)
    print('the predicted number of emails sent by sara is ', cnt_array[0])
    print('the predicted number of emails sent by chris is ', cnt_array[1])
    # who's mail
    # print('prediction of No.10 is ',check_name(pred[10]))
    # print('prediction of No.26 is ',check_name(pred[26]))
    # print('prediction of No.50 is ',check_name(pred[50]))

    # output the image
    # prettyPicture(clf,features_test,labels_test,'test1.png')

    # # for C = 100, to compare C influence
    # clf = SVC(kernel='rbf', C=100)
    # t = time()
    # clf.fit(features_train, labels_train)
    # print('training time is ', round(time() - t, 3), 's')
    #
    # t = time()
    # pred = clf.predict(features_test)
    # print('predicting time is ', round(time() - t, 3), 's')
    #
    # t = time()
    # acc = accuracy_score(pred, labels_test)
    # print('accuracy time is ', round(time() - t, 3), 's')
    # # another way, it combines prediction and accuracy calculation
    # # acc = clf.score(features_test,labels_test)
    #
    # print('acc = ', round(acc, 3))
    #
    # # output the image
    # prettyPicture(clf, features_test, labels_test, 'test100.png')
#########################################################


