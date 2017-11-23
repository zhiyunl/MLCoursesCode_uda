#!/usr/bin/python

import matplotlib.pyplot as plt

from choose_your_own.class_vis import prettyPicture
from choose_your_own.prep_terrain_data import makeTerrainData

features_train, labels_train, features_test, labels_test = makeTerrainData()


def choose_main():
    ### the training data (features_train, labels_train) have both "fast" and "slow"
    ### points mixed together--separate them so we can give them different colors
    ### in the scatterplot and identify them visually
    grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
    bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
    grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 1]
    bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 1]

    #### initial visualization
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.scatter(bumpy_fast, grade_fast, color="b", label="fast")
    plt.scatter(grade_slow, bumpy_slow, color="r", label="slow")
    plt.legend()
    plt.xlabel("bumpiness")
    plt.ylabel("grade")
    plt.show()
    ################################################################################


    ### your code here!  name your classifier object clf if you want the
    ### visualization code (prettyPicture) to show you the decision boundary

    ## knn
    ## finally we got the best knn parameters are
    ## n_neighbors = 8 or 22, weights = 'uniform'
    ## and the accuracy is 0.944
    # from choose_your_own.knn import knn_main
    # clf = knn_main()

    ## adaboost
    ## n_estimatos ranged from 13 to 24, both get a 0.928 accuracy
    from choose_your_own.adaboost import ada_main
    clf = ada_main()

    try:
        prettyPicture(clf, features_test, labels_test)
    except NameError:
        pass
