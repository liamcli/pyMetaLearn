import numpy as np
import os
import sys
import time
import re

import arff

import sklearn
import sklearn.datasets
import sklearn.utils
import sklearn.cross_validation
import sklearn.svm

import HPOlib.benchmark_util as benchmark_util
import HPOlib.data_util as data_util


# Specify the size of the kernel test_cache (in MB)
SVM_CACHE_SIZE = 2000

ARFF = 0
NPY = 1


"""Hyperparameter values to search in...

Matthias Reif et al.(Meta-learning for evolutionary parameter
optimization of
classifiers uses gamma [0.0001, 10] with 85 logarithmic steps and C [0,
1000] with 100 logarithmic steps.

The practical guide of the LibSVM homepage suggests to use a grid of C =
2^-5, 2^-3,..., 2^-15 and gamma = 2^-15, 2^-13, ..., 2^3.

Gomes et al.(Combining meta-learning and search techniques to select
parameters for support vector machines) follow the guidelines of the
LibSVM authors.
"""


def data_has_categorical_values(X):
    if X.dtype == np.float64 or X.dtype == np.float32:
        return False
    else:
        raise NotImplementedError()


def data_is_normalized(X):
    if np.any(np.min(X)) < 0 or np.any(np.max(X)) > 1:
        return False
    return True


def data_has_missing_values(X):
    return not np.isfinite(X).all()


def normalize(X):
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    return (X - X_min) / (X_max - X_min)


def impute_missing_values(data, strategy='mean'):
    if strategy != 'mean':
        raise NotImplementedError()
    pass


def split_data(X, Y, fold, folds):
    assert fold < folds
    # do stratified cross validation, like OpenML does according to the MySQL
    # dump.
    kf = sklearn.cross_validation.StratifiedKFold(Y, n_folds=folds,
        indices=True)
    for idx, split in enumerate(kf):
        if idx == fold:
            return split


def load_data():
    X_train = data_util.load_file("../train.npy", "numpy", 100)
    Y_train = data_util.load_file("../train_targets.npy", "numpy", 100)
    X_test = data_util.load_file("../test.npy", "numpy", 100)
    Y_test = data_util.load_file("../test_targets.npy", "numpy", 100)
    return X_train, Y_train, X_test, Y_test


def fit(params, data):
    C = 2.**(float(params["C"]))
    gamma = 2.**(float(params["gamma"]))

    #print "C: 2^%f=%f, gamma: 2^%f=%f" % (int(float(params["C"])), C,
    #                                int(float(params["gamma"])), gamma)

    random_state = sklearn.utils.check_random_state(42)
    svm = sklearn.svm.SVC(cache_size=SVM_CACHE_SIZE, C=C, gamma=gamma,
                          kernel="rbf", random_state=random_state)
    print data["train_X"].shape, data["train_Y"].shape
    svm.fit(data["train_X"], data["train_Y"])
    predictions = svm.predict(data["valid_X"])
    accuracy = sklearn.metrics.accuracy_score(data["valid_Y"], predictions)
    # print sklearn.metrics.classification_report(data["test_Y"], predictions)
    # maybe build a classification report
    return accuracy


def main(params, **kwargs):
    fold = int(kwargs["fold"])
    folds = int(kwargs["folds"])

    # kwargs["iris_data"] = sklearn.datasets.load_iris()

    if "iris_data" in kwargs:
        iris_data = kwargs["iris_data"]

        X = iris_data["data"]
        Y = iris_data["target"]

    elif "data_files" in kwargs:
        raise NotImplementedError()

    else:
        X_train, Y_train, X_test,  Y_test = load_data()

    if data_has_categorical_values(X_train) or data_has_categorical_values(X_test):
        raise NotImplementedError()
    if not data_is_normalized(X_train) or not data_is_normalized(X_test):
        raise NotImplementedError()
    if data_has_missing_values(X_train) or data_has_missing_values(X_test):
        raise NotImplementedError()

    train_mask, valid_mask = split_data(X_train, Y_train, fold, folds)
    data = dict()
    data["train_X"] = X_train[train_mask]
    data["train_Y"] = Y_train[train_mask]
    data["valid_X"] = X_train[valid_mask]
    data["valid_Y"] = Y_train[valid_mask]
    data["test_X"] = X_test
    data["test_Y"] = Y_test

    result = fit(params, data)
    return 1 - result


if __name__ == "__main__":
    if len(sys.argv) == 1:
        iris_data = sklearn.datasets.load_iris()
        results = []
        for fold in range(10):
            results.append(main({"C": "0", "gamma": "-2"}, fold=fold,
                                folds=10, iris_data=iris_data))
        print results
        print np.mean(results), np.std(results)

    else:
        starttime = time.time()
        args, params = benchmark_util.parse_cli()
        print params
        result = main(params, **args)
        duration = time.time() - starttime
        print "Result for ParamILS: %s, %f, 1, %f, %d, %s" % \
            ("SAT", abs(duration), result, -1, str(__file__))
