from argparse import ArgumentParser
import cPickle
import itertools
import numpy as np
import os
import pandas as pd
import scipy.stats


import pyMetaLearn.optimizers.metalearn_optimizer.metalearner as metalearner


def create_predict_spearman_rank(metafeatures, experiments):
    X = []
    Y = []
    # Calculate the pairwise ranks between datasets
    dataset_names = [name for name in metafeatures.index]
    cross_product = []
    for cross in itertools.combinations_with_replacement(dataset_names, r=2):
        cross_product.append(cross)
    print "Using %d datasets" % len(dataset_names)
    print "This will results in %d training points" % len(cross_product)

    # Create inputs and targets
    for cross in cross_product:
        mf_1 = metafeatures.loc[cross[0]]
        mf_2 = metafeatures.loc[cross[1]]
        assert mf_1.dtype == np.float64
        assert mf_2.dtype == np.float64
        x = np.hstack((mf_1, mf_2))
        pd.Series(data=x, index=(cross[0], cross[1]))
        X.append(x)

        experiments_1 = experiments[cross[0]]
        experiments_2 = experiments[cross[1]]
        assert len(experiments_1) == len(experiments_2)

        responses_1 = np.zeros((len(experiments_1)), dtype=np.float64)
        responses_2 = np.zeros((len(experiments_1)), dtype=np.float64)
        for idx, zipped in enumerate(zip(experiments_1, experiments_2)):
            # Test if the order of the params is the same
            exp_1, exp_2 = zipped
            assert exp_1.params == exp_2.params
            responses_1[idx] = exp_1.result
            responses_2[idx] = exp_2.result

        rho, p = scipy.stats.spearmanr(responses_1, responses_2)
        Y.append(rho)
    X = np.array(X)
    Y = np.array(Y)
    print "Metafeatures", metafeatures.shape
    print "X", X.shape
    print "Y", Y.shape
    assert X.shape == (len(cross_product), metafeatures.shape[1] * 2), \
        (X.shape, (len(cross), metafeatures.shape[1] * 2))
    assert Y.shape == (len(cross_product), )
    # train sklearn regressor (tree) with 10fold CV
    indices = range(len(X))
    np_rs = np.random.RandomState(42)
    np_rs.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    return X, Y


def create_predict_spearman_rank_with_cv(metafeatures, experiments):
    X = []
    Y = []
    # Calculate the pairwise ranks between datasets
    dataset_names = [name for name in metafeatures.index]
    cross_product = []
    for cross in itertools.combinations_with_replacement(dataset_names, r=2):
        cross_product.append(cross)

    print "Using %d datasets" % len(dataset_names)
    print "This will results in %d training points" % len(cross_product)

    # Create inputs and targets
    for cross in cross_product:
        mf_1 = metafeatures.loc[cross[0]]
        mf_2 = metafeatures.loc[cross[1]]
        assert mf_1.dtype == np.float64
        assert mf_2.dtype == np.float64
        x = np.hstack((mf_1, mf_2))
        X.append(x)

        experiments_1 = experiments[cross[0]]
        experiments_2 = experiments[cross[1]]
        assert len(experiments_1) == len(experiments_2)

        responses_1 = np.zeros((len(experiments_1)), dtype=np.float64)
        responses_2 = np.zeros((len(experiments_1)), dtype=np.float64)
        for idx, zipped in enumerate(zip(experiments_1, experiments_2)):
            # Test if the order of the params is the same
            exp_1, exp_2 = zipped
            assert exp_1.params == exp_2.params
            responses_1[idx] = exp_1.result
            responses_2[idx] = exp_2.result

        rho, p = scipy.stats.spearmanr(responses_1, responses_2)
        Y.append(rho)
    X = np.array(X)
    Y = np.array(Y)
    print "Metafeatures", metafeatures.shape
    print "X", X.shape
    print "Y", Y.shape
    assert X.shape == (len(cross_product), metafeatures.shape[1] * 2), \
        (X.shape, (len(cross), metafeatures.shape[1] * 2))
    assert Y.shape == (len(cross_product), )
    # train sklearn regressor (tree) with 10fold CV
    indices = range(len(X))
    np_rs = np.random.RandomState(42)
    np_rs.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    return X, Y


if __name__ == "__main__":
    # TODO: right now, this is only done for one split, namely the split of
    # the directory we're inside...
    parser = ArgumentParser()
    parser.add_argument("target_directory", type=str)
    args = parser.parse_args()

    target_directory = args.target_directory
    if not os.path.exists(target_directory):
        raise ValueError("Target directory %s does not exist." % target_directory)

    # Important, change into some directory in which an experiment was already
    # performed...
    context = metalearner.setup(None)
    metafeatures = context["metafeatures"]
    meta_base = context["meta_base"]

    # Experiment is an OrderedDict, which has dataset names as keys
    # The values are lists of experiments(OrderedDict of params, response)
    experiments = meta_base.experiments

    X, Y = create_predict_spearman_rank(metafeatures, experiments)
    spearman_rank_file = os.path.join(target_directory, "spearman_rank.pkl")
    with open(spearman_rank_file, "w") as fh:
        cPickle.dump((X, Y, metafeatures), fh)

    # Calculate the metafeatures for the 10fold CV...
    X, Y = create_predict_spearman_rank_with_cv(metafeatures, experiments)
    spearman_rank_file = os.path.join(target_directory, "spearman_rank_cv.pkl")
    with open(spearman_rank_file, "w") as fh:
        cPickle.dump((X, Y, metafeatures), fh)


