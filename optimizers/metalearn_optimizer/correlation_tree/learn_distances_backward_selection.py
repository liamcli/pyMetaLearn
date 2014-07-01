from argparse import ArgumentParser
from collections import defaultdict
import cPickle
import numpy as np
import os
import scipy.stats
import sys
import time

import sklearn.cross_validation
import sklearn.ensemble
import sklearn.utils
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.svm

import pyMetaLearn.openml.manage_openml_data
import HPOlib.benchmark_util as benchmark_util


def split_for_loo(X, Y, dataset_name):
    train = []
    valid = []
    for cross in X.index:
        if dataset_name in cross:
            valid.append(cross)
        else:
            train.append(cross)

    X_train = X.loc[train].values
    Y_train = Y.loc[train].values
    X_valid = X.loc[valid].values
    Y_valid = Y.loc[valid].values
    # print X_train.shape, Y_train.shape, X_valid.shape, Y_valid.shape
    return X_train, Y_train, X_valid, Y_valid


def train_model_without_one_dataset(X, Y, model, dataset_name):
    X_train, Y_train, X_valid, Y_valid = split_for_loo(X, Y, dataset_name)
    model.fit(X_train, Y_train)
    return model


def validate_model_without_one_dataset(X, Y, model, dataset_name):
    X_train, Y_train, X_valid, Y_valid = split_for_loo(X, Y, dataset_name)
    predictions = model.predict(X_valid)
    rho =  scipy.stats.kendalltau(Y_valid, predictions)[0]
    mae = sklearn.metrics.mean_absolute_error(predictions, Y_valid)
    mse = sklearn.metrics.mean_squared_error(predictions, Y_valid)

    return mae, mse, rho


def get_rf(n_estimators=10, max_features=1.0, min_samples_split=2,
           min_samples_leaf=1, n_jobs=2, seed=42):
    rs = sklearn.utils.check_random_state(int(seed))
    rf = sklearn.ensemble.RandomForestRegressor(n_estimators=int(n_estimators),
                                                max_features=float(max_features),
                                                min_samples_split=int(min_samples_split),
                                                min_samples_leaf=int(min_samples_leaf),
                                                criterion="mse",
                                                random_state=rs,
                                                n_jobs=int(n_jobs))
    return rf


def train_rf_without_one_dataset(X, Y, dataset_name, params, save_name=None):
    local_directory = pyMetaLearn.openml.manage_openml_data.get_local_directory()
    model_directory = os.path.join(local_directory, "models_distances")
    if sys.maxsize > 2**32:
        model_directory += "_64"

    if not os.path.exists(model_directory):
        os.mkdir(model_directory)

    if save_name is not None:
        save_name += ".pkl"

    if save_name is None:
        rf = get_rf(**params)
        rf = train_model_without_one_dataset(X, Y, rf, dataset_name)
    elif not os.path.exists(os.path.join(model_directory, save_name)):
        rf = get_rf(**params)
        rf = train_model_without_one_dataset(X, Y, rf, dataset_name)
        with open(os.path.join(model_directory, save_name), "w") as fh:
            cPickle.dump(rf, fh, -1)
    else:
        with open(os.path.join(model_directory, save_name)) as fh:
            rf = cPickle.load(fh)

    return rf


def train_rf(X, Y, metafeatures, params, save_name=None):
    print "Metafeatures", metafeatures.shape
    print "X", X.shape
    print "Y", Y.shape

    metafeature_ranks = defaultdict(float)
    loo_mae = []
    loo_rho = []
    loo_mse = []

    print "Dataset Mae MSE Rho"
    # Leave one out CV
    for idx in range(metafeatures.shape[0]):
        leave_out_dataset = metafeatures.index[idx]

        if save_name is not None:
            tmp_save_name = save_name + leave_out_dataset
        else:
            tmp_save_name = None

        rf = train_rf_without_one_dataset(X, Y, leave_out_dataset, params,
                                          save_name=tmp_save_name)
        mae, mse, rho = validate_model_without_one_dataset(X, Y, rf,
                                                    leave_out_dataset)
        print leave_out_dataset, mae, mse, rho

        loo_mae.append(mae)
        loo_rho.append(rho)
        loo_mse.append(mse)
        mf_importances = [(rf.feature_importances_[i], X.columns[i]) for i
                           in range(X.shape[1])]
        mf_importances.sort()
        mf_importances.reverse()
        for rank, item in enumerate(mf_importances):
            score, mf_name = item
            metafeature_ranks[mf_name] += float(rank)

    mae = np.mean(loo_mae)
    mae_std = np.std(loo_mae)
    mse = np.mean(loo_mse)
    mse_std = np.mean(loo_mse)
    rho = np.mean(loo_rho)
    rho_std = np.std(loo_rho)

    mean_ranks = [(metafeature_ranks[mf_name] / metafeatures.shape[0], mf_name)
                  for mf_name in X.columns]
    mean_ranks.sort()

    return mae, mae_std, mse, mse_std, rho, rho_std, mean_ranks


if __name__ == "__main__":
    starttime = time.time()
    print sys.argv
    args, params = benchmark_util.parse_cli()

    input_file = args["input_file"]
    if not os.path.exists(input_file):
        raise ValueError("Input file %s does not exist." %
                         input_file)
    save_file_prefix = "rf_" + os.path.split(input_file)[1].split(".")[0] + "_"

    with open(input_file) as fh:
        X, Y, metafeatures = cPickle.load(fh)

    if not all(np.isfinite(X)):
        print "X is not finite"
        exit(1)
    if not all(np.isfinite(Y)):
        print "Y is not finite"
        exit(1)

    last_mae = 100.
    eliminate = set()
    while True:
        print "###############################################################"
        keep = [col for col in X.columns if col not in eliminate]
        X_subset = X.loc[:,keep]

        mae, mae_std, mse, mse_std, rho, rho_std, mean_ranks = \
            train_rf(X_subset, Y, metafeatures, params)

        print "MAE", mae, mae_std
        print "MSE", mse, mse_std
        print "Mean tau", rho, rho_std
        for rank in mean_ranks:
            print rank

        # if mae > last_mae:
        #     break
        last_mae = mae

        # Eliminate two metafeatures
        eliminate.add(mean_ranks[-1][1])
        eliminate.add(mean_ranks[-2][1])

        print "Eliminate features %s %s" % (mean_ranks[-1][1], mean_ranks[-2][1])
        print eliminate

    # TODO: save only validate-best runs!

    duration = time.time() - starttime
    print "Result for ParamILS: %s, %f, 1, %f, %d, %s" % \
        ("SAT", abs(duration), mae, -1, str(__file__))