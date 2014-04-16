import numpy as np
import os
import sys
import time

import sklearn
import sklearn.datasets
import sklearn.utils
import sklearn.cross_validation as cross_validation
import sklearn.svm

import HPOlib.benchmark_util as benchmark_util
import HPOlib.wrapping_util as wrapping_util

from pyMetaLearn.dataset_base import DatasetBase
from pyMetaLearn.openml.openml_task import OpenMLTask


# Specify the size of the kernel test_cache (in MB)
SVM_CACHE_SIZE = 2000


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


def task_evaluate(params, fold=0, folds=1):
    random_state = sklearn.utils.check_random_state(42)
    C = 2.**(float(params["C"]))
    gamma = 2.**(float(params["gamma"]))
    algo = sklearn.svm.SVC(cache_size=SVM_CACHE_SIZE, C=C, gamma=gamma,
                           kernel="rbf" , random_state=random_state)

    config = wrapping_util.load_experiment_config_file()
    task_id, task_type, data_set_id, target_feature,
    estimation_procudure_type, data_splits_url,
    estimation_parameters, evaluation_measure
    task = OpenMLTask()
    X, Y = task.get_dataset()

    if data_has_categorical_values(X):
        raise NotImplementedError()
    if not data_is_normalized(X):
        raise NotImplementedError()
    if data_has_missing_values(X):
        raise NotImplementedError()

    if isinstance(Y.dtype, float):
        raise ValueError("SVC is used for classification, the target "
                         "values are float values which implies this is a "
                         "regression task.")
    elif not isinstance(Y.dtype, int):
        raise NotImplementedError(Y.dtype)

    accuracy = task.partial_evaluate(algo, fold, folds)
    return 1 - accuracy


if __name__ == "__main__":
    starttime = time.time()
    args, params = benchmark_util.parse_cli()
    print params
    result = task_evaluate(params, **args)
    duration = time.time() - starttime
    print "Result for ParamILS: %s, %f, 1, %f, %d, %s" % \
        ("SAT", abs(duration), result, -1, str(__file__))
