from collections import OrderedDict
import cPickle
import numpy as np
import numpy.ma as ma
import sys
import time

import sklearn
import sklearn.datasets
import sklearn.utils
import sklearn.cross_validation as cross_validation
import sklearn.svm

import HPOlib.benchmark_util as benchmark_util
import HPOlib.wrapping_util as wrapping_util


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


def get_fold(X, Y, fold, folds):
    assert fold < folds
    # do stratified cross validation, like OpenML does according to the MySQL
    # dump.
    print fold, "/", folds
    kf = cross_validation.StratifiedKFold(Y, n_folds=folds, indices=True)
    for idx, split in enumerate(kf):
        if idx == fold:
            return split


def fit(params, data):
    C = 2.**(float(params["C"]))
    gamma = 2.**(float(params["gamma"]))

    random_state = sklearn.utils.check_random_state(42)
    svm = sklearn.svm.SVC(cache_size=SVM_CACHE_SIZE, C=C, gamma=gamma,
                          kernel="rbf", random_state=random_state)
    svm.fit(data["train_X"], data["train_Y"])

    predictions = svm.predict(data["valid_X"])
    accuracy = sklearn.metrics.accuracy_score(data["valid_Y"], predictions)
    return accuracy


def convert_pandas_to_npy(X):
    """Nominal values are replaced with a one hot encoding and missing
     values represented with zero."""
    num_fields = 0
    attribute_arrays = []
    keys = []

    for idx, attribute in enumerate(X.iteritems()):
        attribute_name = attribute[0].lower()
        attribute_type = attribute[1].dtype
        row = attribute[1]

        if attribute_type == np.float64:
            rval = _parse_numeric(row)
            if rval is not None:
                keys.append(attribute_name)
                attribute_arrays.append(rval)
                num_fields += 1

        elif attribute_type == 'object':
            rval = _parse_nominal(row)
            if rval is not None:
                attribute_arrays.append(rval)
                num_fields += rval.shape[1]
                if rval.shape[1] == 1:
                    keys.append(attribute_name)
                else:
                    vals = [attribute_name + ":" + str(possible_value) for
                            possible_value in range(rval.shape[1])]
                    keys.extend(vals)

        else:
            raise NotImplementedError()

    dataset_array = np.ndarray((X.shape[0], num_fields))
    col_idx = 0
    for attribute_array in attribute_arrays:
        print attribute_array.shape
        length = attribute_array.shape[1]
        dataset_array[:, col_idx:col_idx + length] = attribute_array
        col_idx += length
    return dataset_array


def encode_labels(row):
    discrete_values = set(row)
    discrete_values.discard(None)
    discrete_values.discard(np.NaN)
    # Adds reproduceability over multiple systems
    discrete_values = sorted(discrete_values)
    encoding = OrderedDict()
    for row_idx, possible_value in enumerate(discrete_values):
        encoding[possible_value] = row_idx
    return encoding


def _parse_nominal(row):
    # This few lines perform a OneHotEncoding, where missing
    # values represented by none of the attributes being active (
    # a feature which i could not implement with sklearn).
    # Different imputation strategies can easily be added by
    # extracting a method from the else clause.
    # Caution: this methodology only keeps values that are
    # encountered in the dataset. If this is a subset of the
    # possible values of the arff file, only the subset is
    # encoded via the OneHotEncoding
    encoding = encode_labels(row)

    if len(encoding) == 0:
        return None

    array = np.zeros((row.shape[0], len(encoding)))

    for row_idx, value in enumerate(row):
        if row[row_idx] is not None:
            array[row_idx][encoding[row[row_idx]]] = 1

    return array


def _parse_numeric(row):
    # NaN and None will be treated as missing values
    array = np.array(row).reshape((-1, 1))

    if not np.any(np.isfinite(array)):
        return None

    # Apply scaling here so that if we are setting missing values
    # to zero, they are still zero afterwards
    X_min = np.nanmin(array, axis=0)
    X_max = np.nanmax(array, axis=0)
    # Numerical stability...
    if (X_max - X_min) > 0.0000000001:
        array = (array - X_min) / (X_max - X_min)

    # Replace invalid values (~np.isfinite)
    fixed_array = ma.fix_invalid(array, copy=True, fill_value=0)

    if not np.isfinite(fixed_array).all():
        print fixed_array
        raise NotImplementedError()

    return fixed_array


def main(params, **kwargs):
    fold = int(kwargs["fold"])
    folds = int(kwargs["folds"])

    if ("dataset_file" in kwargs and "test_folds" in kwargs and "test_fold" in kwargs):
        test_fold = int(kwargs["test_fold"])
        test_folds = int(kwargs["test_folds"])
        dataset_file = kwargs["dataset_file"]
    else:
        config = wrapping_util.load_experiment_config_file()
        test_fold = config.getint("EXPERIMENT", "test_fold")
        test_folds = config.getint("EXPERIMENT", "test_folds")
        dataset_file = config.get("EXPERIMENT", "dataset")

    # 4. Do column-wise pre-processing and insert the values into the numpy
    #    array
    # 5. Run the libSVM

    # kwargs["iris_data"] = sklearn.datasets.load_iris()

    if "iris_data" in kwargs:
        iris_data = kwargs["iris_data"]

        X = iris_data["data"]
        Y = iris_data["target"]

    elif "data_files" in kwargs:
        raise NotImplementedError()

    else:
        with open(dataset_file) as fh:
            dataset = cPickle.load(fh)
        X, Y = dataset.get_processed_files()
        X = convert_pandas_to_npy(X)
        if Y.dtype == np.float64:
            raise ValueError("SVC is used for classification, the target "
                             "values are float values which implies this is a "
                             "regression task")
        elif Y.dtype == 'object':
            encoding = encode_labels(Y)
            Y = np.array([encoding[value] for value in Y])
        else:
            raise NotImplementedError(Y.dtype)

    rs = np.random.RandomState(42)
    indices = np.arange(X.shape[0])
    rs.shuffle(indices)
    X = X[indices]
    Y = Y[indices]

    print "Loaded Data X %s and Y %s" % (str(X.shape), str(Y.shape))

    # Split the dataset according to test_fold and test_folds
    split = get_fold(X, Y, fold=test_fold, folds=test_folds)
    X_train = X[split[0]]
    X_test = X[split[1]]
    Y_train = Y[split[0]]
    Y_test = Y[split[1]]
    print Y_train

    if data_has_categorical_values(X_train) or data_has_categorical_values(X_test):
        raise NotImplementedError()
    if not data_is_normalized(X_train) or not data_is_normalized(X_test):
        raise NotImplementedError()
    if data_has_missing_values(X_train) or data_has_missing_values(X_test):
        raise NotImplementedError()

    train_mask, valid_mask = get_fold(X_train, Y_train, fold, folds)
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
