#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys

sys.path.append("../tools/")
from tools.email_preprocess import preprocess
from sklearn import tree

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


# 1 percent of data
# features_train = features_train[:int(len(features_train) / 100)]
# labels_train = labels_train[:int(len(labels_train) / 100)]


#########################################################
### your code goes here ###
def dt_main():
    clf = tree.DecisionTreeClassifier(min_samples_split=40)

    clf.fit(features_train, labels_train)

    acc = clf.score(features_test, labels_test)

    print("acc = ", round(acc, 3))

    features_num = len(features_train[0])
    print('number of features is ', features_num)
#########################################################


