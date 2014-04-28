from argparse import ArgumentParser
import cPickle
import numpy as np
import os

import sklearn.cross_validation
import sklearn.ensemble
import sklearn.utils
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.svm


def train_gps(X, Y, metafeatures):
    pass


def train_rf(X, Y, metafeatures):
    print "Metafeatures", metafeatures.shape
    print "X", X.shape
    print "Y", Y.shape

    # Maybe leave out some attributes...
    # keep = ([True if column != "class_probability_max" else False for
    #            column in metafeatures.columns])
    # keep = np.array(keep + keep)

    """
    loo_mae = 0.
    loo_mse = 0.

    # Leave one out CV
    for idx in range(metafeatures.shape[0]):
        train = []
        test = []

        print metafeatures.index[test]
        rs = sklearn.utils.check_random_state(42)
        rf = sklearn.ensemble.RandomForestRegressor(n_estimators=100,
                                                     criterion="mse",
                                                     random_state=rs)
        rf.fit(X[train], Y[train])
        predictions = rf.predict(X[test])
        loo_mae += sklearn.metrics.mean_absolute_error(predictions, Y[test])
        loo_mse += sklearn.metrics.mean_squared_error(predictions, Y[test])
        print sklearn.metrics.mean_absolute_error(predictions, Y[test])

    print "LOO MAE", loo_mae / metafeatures.shape[0]
    print "LOO MSE", loo_mse / metafeatures.shape[0]
    """

    cv_mae = 0.
    cv_mse = 0.

    kf = sklearn.cross_validation.KFold(n=X.shape[0], n_folds=10)

    # 10fold CV
    for train, test in kf:

        rs = sklearn.utils.check_random_state(42)
        rf = sklearn.ensemble.RandomForestRegressor(n_estimators=100,
                                                     criterion="mse",
                                                     random_state=rs)
        rf.fit(X[train], Y[train])
        predictions = rf.predict(X[test])
        cv_mae += sklearn.metrics.mean_absolute_error(predictions, Y[test])
        cv_mse += sklearn.metrics.mean_squared_error(predictions, Y[test])
        print sklearn.metrics.mean_absolute_error(predictions, Y[test])

    print "10fold cv MAE", cv_mae / 10
    print "10fold cv MSE", cv_mse / 10

    rs = sklearn.utils.check_random_state(42)
    rf = sklearn.ensemble.RandomForestRegressor(n_estimators=100,
                                                 criterion="mse",
                                                 random_state=rs)
    rf.fit(X, Y)
    print
    print "Feature importances"
    for i, column in enumerate(metafeatures.columns):
        print column, rf.feature_importances_[i], rf.feature_importances_[i + len(metafeatures.columns)]

    # Evaluation protocol...kendalls tau?
    # do the above in leave one out fashion and compare the obtained rankings to
    # the original rankings


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input_file", type=str)
    args = parser.parse_args()

    input_file = args.input_file
    if not os.path.exists(input_file):
        raise ValueError("Input directory %s does not exist." %
                         input_file)

    with open(input_file) as fh:
        X, Y, metafeatures = cPickle.load(fh)

    train_rf(X, Y, metafeatures)