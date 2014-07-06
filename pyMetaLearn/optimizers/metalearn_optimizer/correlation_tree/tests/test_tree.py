"""
Testing for the tree module (sklearn.tree).
"""
import pickle
import numpy as np

from itertools import product

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import assert_less

from pyMetaLearn.optimizers.metalearn_optimizer.correlation_tree.tree import \
    DecisionTreeClassifier
from pyMetaLearn.optimizers.metalearn_optimizer.correlation_tree.tree import \
    DecisionTreeRegressor
from pyMetaLearn.optimizers.metalearn_optimizer.correlation_tree.tree import \
    ExtraTreeClassifier
from pyMetaLearn.optimizers.metalearn_optimizer.correlation_tree.tree import \
    ExtraTreeRegressor

from sklearn import tree
from sklearn import datasets
from sklearn.utils.fixes import bincount

from sklearn.preprocessing import balance_weights


CLF_CRITERIONS = ("gini", "entropy")
REG_CRITERIONS = ("mse", "kendall")

CLF_TREES = {
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "ExtraTreeClassifier": ExtraTreeClassifier,
}

REG_TREES = {
    "DecisionTreeRegressor": DecisionTreeRegressor,
    "ExtraTreeRegressor": ExtraTreeRegressor,
}

ALL_TREES = dict()
ALL_TREES.update(CLF_TREES)
ALL_TREES.update(REG_TREES)


# toy sample
X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
y = [-1, -1, -1, 1, 1, 1]
T = [[-1, -1], [2, 2], [3, 2]]
true_result = [-1, 1, 1]

# also load the iris dataset
# and randomly permute it
iris = datasets.load_iris()
rng = np.random.RandomState(1)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]

# also load the boston dataset
# and randomly permute it
boston = datasets.load_boston()
perm = rng.permutation(boston.target.size)
boston.data = boston.data[perm]
boston.target = boston.target[perm]


def test_classification_toy():
    """Check classification on a toy dataset."""
    for name, Tree in CLF_TREES.items():
        clf = Tree(random_state=0)
        clf.fit(X, y)
        assert_array_equal(clf.predict(T), true_result,
                           "Failed with {0}".format(name))

        clf = Tree(max_features=1, random_state=1)
        clf.fit(X, y)
        assert_array_equal(clf.predict(T), true_result,
                           "Failed with {0}".format(name))


def test_weighted_classification_toy():
    """Check classification on a weighted toy dataset."""
    for name, Tree in CLF_TREES.items():
        clf = Tree(random_state=0)

        clf.fit(X, y, sample_weight=np.ones(len(X)))
        assert_array_equal(clf.predict(T), true_result,
                           "Failed with {0}".format(name))

        clf.fit(X, y, sample_weight=np.ones(len(X)) * 0.5)
        assert_array_equal(clf.predict(T), true_result,
                           "Failed with {0}".format(name))


def test_regression_toy():
    """Check regression on a toy dataset."""
    for name, Tree in REG_TREES.items():
        reg = Tree(random_state=1)
        reg.fit(X, y)
        assert_almost_equal(reg.predict(T), true_result,
                            err_msg="Failed with {0}".format(name))

        clf = Tree(max_features=1, random_state=1)
        clf.fit(X, y)
        assert_almost_equal(reg.predict(T), true_result,
                            err_msg="Failed with {0}".format(name))


def test_kendalls_tau():
    reference = DecisionTreeRegressor(criterion="kendall", random_state=1)
    reference.fit(boston.data[:400], boston.target[:400])
    reference_prediction = reference.predict(boston.data[400:])

    for idx in range(5):
        reg = DecisionTreeRegressor(criterion="kendall", random_state=1)
        reg.fit(boston.data[:400], boston.target[:400])
        pred = reg.predict(boston.data[400:])
        # print "%d: %s, %s" % \
        #     (idx, reference.tree_.children_left, reg.tree_.children_left)
        assert_almost_equal(pred, reference_prediction, )


def test_xor():
    """Check on a XOR problem"""
    y = np.zeros((10, 10))
    y[:5, :5] = 1
    y[5:, 5:] = 1

    gridx, gridy = np.indices(y.shape)

    X = np.vstack([gridx.ravel(), gridy.ravel()]).T
    y = y.ravel()

    for name, Tree in CLF_TREES.items():
        clf = Tree(random_state=0)
        clf.fit(X, y)
        assert_equal(clf.score(X, y), 1.0,
                     "Failed with {0}".format(name))

        clf = Tree(random_state=0, max_features=1)
        clf.fit(X, y)
        assert_equal(clf.score(X, y), 1.0,
                     "Failed with {0}".format(name))


def test_iris():
    """Check consistency on dataset iris."""
    for (name, Tree), criterion in product(CLF_TREES.items(), CLF_CRITERIONS):
        clf = Tree(criterion=criterion, random_state=0)
        clf.fit(iris.data, iris.target)
        score = accuracy_score(clf.predict(iris.data), iris.target)
        assert_greater(score, 0.9,
                       "Failed with {0}, criterion = {1} and score = {2}"
                       "".format(name, criterion, score))

        clf = Tree(criterion=criterion, max_features=2, random_state=0)
        clf.fit(iris.data, iris.target)
        score = accuracy_score(clf.predict(iris.data), iris.target)
        assert_greater(score, 0.5,
                       "Failed with {0}, criterion = {1} and score = {2}"
                       "".format(name, criterion, score))


def test_boston():
    """Check consistency on dataset boston house prices."""

    for (name, Tree), criterion in product(REG_TREES.items(), REG_CRITERIONS):
        reg = Tree(criterion=criterion, random_state=0)
        reg.fit(boston.data, boston.target)
        score = mean_squared_error(boston.target, reg.predict(boston.data))
        assert_less(score, 1,
                    "Failed with {0}, criterion = {1} and score = {2}"
                    "".format(name, criterion, score))

        # using fewer features reduces the learning ability of this tree,
        # but reduces training time.
        reg = Tree(criterion=criterion, max_features=6, random_state=0)
        reg.fit(boston.data, boston.target)
        score = mean_squared_error(boston.target, reg.predict(boston.data))
        assert_less(score, 2,
                    "Failed with {0}, criterion = {1} and score = {2}"
                    "".format(name, criterion, score))


def test_probability():
    """Predict probabilities using DecisionTreeClassifier."""

    for name, Tree in CLF_TREES.items():
        clf = Tree(max_depth=1, max_features=1, random_state=42)
        clf.fit(iris.data, iris.target)

        prob_predict = clf.predict_proba(iris.data)
        assert_array_almost_equal(np.sum(prob_predict, 1),
                                  np.ones(iris.data.shape[0]),
                                  err_msg="Failed with {0}".format(name))
        assert_array_equal(np.argmax(prob_predict, 1),
                           clf.predict(iris.data),
                           err_msg="Failed with {0}".format(name))
        assert_almost_equal(clf.predict_proba(iris.data),
                            np.exp(clf.predict_log_proba(iris.data)), 8,
                            err_msg="Failed with {0}".format(name))


def test_arrayrepr():
    """Check the array representation."""
    # Check resize
    X = np.arange(10000)[:, np.newaxis]
    y = np.arange(10000)

    for name, Tree in REG_TREES.items():
        reg = Tree(max_depth=None, random_state=0)
        reg.fit(X, y)


def test_pure_set():
    """Check when y is pure."""
    X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
    y = [1, 1, 1, 1, 1, 1]

    for name, TreeClassifier in CLF_TREES.items():
        clf = TreeClassifier(random_state=0)
        clf.fit(X, y)
        assert_array_equal(clf.predict(X), y,
                           err_msg="Failed with {0}".format(name))

    for name, TreeRegressor in REG_TREES.items():
        reg = TreeRegressor(random_state=0)
        reg.fit(X, y)
        assert_almost_equal(clf.predict(X), y,
                            err_msg="Failed with {0}".format(name))


def test_numerical_stability():
    """Check numerical stability."""
    X = np.array([
        [152.08097839, 140.40744019, 129.75102234, 159.90493774],
        [142.50700378, 135.81935120, 117.82884979, 162.75781250],
        [127.28772736, 140.40744019, 129.75102234, 159.90493774],
        [132.37025452, 143.71923828, 138.35694885, 157.84558105],
        [103.10237122, 143.71928406, 138.35696411, 157.84559631],
        [127.71276855, 143.71923828, 138.35694885, 157.84558105],
        [120.91514587, 140.40744019, 129.75102234, 159.90493774]])

    y = np.array(
        [1., 0.70209277, 0.53896582, 0., 0.90914464, 0.48026916,  0.49622521])

    with np.errstate(all="raise"):
        for name, Tree in REG_TREES.items():
            reg = Tree(random_state=0)
            reg.fit(X, y)
            reg.fit(X, -y)
            reg.fit(-X, y)
            reg.fit(-X, -y)


def test_importances():
    """Check variable importances."""
    X, y = datasets.make_classification(n_samples=1000,
                                        n_features=10,
                                        n_informative=3,
                                        n_redundant=0,
                                        n_repeated=0,
                                        shuffle=False,
                                        random_state=0)

    for name, Tree in CLF_TREES.items():
        clf = Tree(random_state=0)
        clf.fit(X, y)
        importances = clf.feature_importances_
        n_important = np.sum(importances > 0.1)

        assert_equal(importances.shape[0], 10, "Failed with {0}".format(name))
        assert_equal(n_important, 3, "Failed with {0}".format(name))

        X_new = clf.transform(X, threshold="mean")
        assert_less(0, X_new.shape[1], "Failed with {0}".format(name))
        assert_less(X_new.shape[1], X.shape[1], "Failed with {0}".format(name))


def test_max_features():
    """Check max_features."""
    for name, TreeRegressor in REG_TREES.items():
        reg = TreeRegressor(max_features="auto")
        reg.fit(boston.data, boston.target)
        assert_equal(reg.splitter_.max_features, boston.data.shape[1])

    for name, TreeClassifier in CLF_TREES.items():
        clf = TreeClassifier(max_features="auto")
        clf.fit(iris.data, iris.target)
        assert_equal(clf.splitter_.max_features, 2)

    for name, TreeEstimator in ALL_TREES.items():
        est = TreeEstimator(max_features="sqrt")
        est.fit(iris.data, iris.target)
        assert_equal(est.splitter_.max_features,
                     int(np.sqrt(iris.data.shape[1])))

        est = TreeEstimator(max_features="log2")
        est.fit(iris.data, iris.target)
        assert_equal(est.splitter_.max_features,
                     int(np.log2(iris.data.shape[1])))

        est = TreeEstimator(max_features=1)
        est.fit(iris.data, iris.target)
        assert_equal(est.splitter_.max_features, 1)

        est = TreeEstimator(max_features=3)
        est.fit(iris.data, iris.target)
        assert_equal(est.splitter_.max_features, 3)

        est = TreeEstimator(max_features=0.5)
        est.fit(iris.data, iris.target)
        assert_equal(est.splitter_.max_features,
                     int(0.5 * iris.data.shape[1]))

        est = TreeEstimator(max_features=1.0)
        est.fit(iris.data, iris.target)
        assert_equal(est.splitter_.max_features, iris.data.shape[1])

        est = TreeEstimator(max_features=None)
        est.fit(iris.data, iris.target)
        assert_equal(est.splitter_.max_features, iris.data.shape[1])

        # use values of max_features that are invalid
        est = TreeEstimator(max_features=10)
        assert_raises(ValueError, est.fit, X, y)

        est = TreeEstimator(max_features=-1)
        assert_raises(ValueError, est.fit, X, y)

        est = TreeEstimator(max_features=0.0)
        assert_raises(ValueError, est.fit, X, y)

        est = TreeEstimator(max_features=1.5)
        assert_raises(ValueError, est.fit, X, y)

        est = TreeEstimator(max_features="foobar")
        assert_raises(ValueError, est.fit, X, y)


def test_error():
    """Test that it gives proper exception on deficient input."""
    for name, TreeEstimator in CLF_TREES.items():
        # predict before fit
        est = TreeEstimator()
        assert_raises(Exception, est.predict_proba, X)

        est.fit(X, y)
        X2 = [-2, -1, 1]  # wrong feature shape for sample
        assert_raises(ValueError, est.predict_proba, X2)

    for name, TreeEstimator in ALL_TREES.items():
        # Invalid values for parameters
        assert_raises(ValueError, TreeEstimator(min_samples_leaf=-1).fit, X, y)
        assert_raises(ValueError, TreeEstimator(min_samples_split=-1).fit,
                      X, y)
        assert_raises(ValueError, TreeEstimator(max_depth=-1).fit, X, y)
        assert_raises(ValueError, TreeEstimator(max_features=42).fit, X, y)

        # Wrong dimensions
        est = TreeEstimator()
        y2 = y[:-1]
        assert_raises(ValueError, est.fit, X, y2)

        # Test with arrays that are non-contiguous.
        Xf = np.asfortranarray(X)
        est = TreeEstimator()
        est.fit(Xf, y)
        assert_almost_equal(est.predict(T), true_result)

        # predict before fitting
        est = TreeEstimator()
        assert_raises(Exception, est.predict, T)

        # predict on vector with different dims
        est.fit(X, y)
        t = np.asarray(T)
        assert_raises(ValueError, est.predict, t[:, 1:])

        # wrong sample shape
        Xt = np.array(X).T

        est = TreeEstimator()
        est.fit(np.dot(X, Xt), y)
        assert_raises(ValueError, est.predict, X)

        clf = TreeEstimator()
        clf.fit(X, y)
        assert_raises(ValueError, clf.predict, Xt)


def test_min_samples_leaf():
    """Test if leaves contain more than leaf_count training examples"""
    X = np.asfortranarray(iris.data.astype(tree._tree.DTYPE))
    y = iris.target

    for name, TreeEstimator in ALL_TREES.items():
        est = TreeEstimator(min_samples_leaf=5, random_state=0)
        est.fit(X, y)
        out = est.tree_.apply(X)
        node_counts = np.bincount(out)
        leaf_count = node_counts[node_counts != 0]  # drop inner nodes
        assert_greater(np.min(leaf_count), 4,
                       "Failed with {0}".format(name))


def test_pickle():
    """Check that tree estimator are pickable """
    for name, TreeClassifier in CLF_TREES.items():
        clf = TreeClassifier(random_state=0)
        clf.fit(iris.data, iris.target)
        score = clf.score(iris.data, iris.target)

        serialized_object = pickle.dumps(clf)
        clf2 = pickle.loads(serialized_object)
        assert_equal(type(clf2), clf.__class__)
        score2 = clf2.score(iris.data, iris.target)
        assert_equal(score, score2, "Failed to generate same score "
                                    "after pickling (classification) "
                                    "with {0}".format(name))

    for name, TreeRegressor in REG_TREES.items():
        reg = TreeRegressor(random_state=0)
        reg.fit(boston.data, boston.target)
        score = reg.score(boston.data, boston.target)

        serialized_object = pickle.dumps(reg)
        reg2 = pickle.loads(serialized_object)
        assert_equal(type(reg2), reg.__class__)
        score2 = reg2.score(boston.data, boston.target)
        assert_equal(score, score2, "Failed to generate same score "
                                    "after pickling (regression) "
                                    "with {0}".format(name))


def test_multioutput():
    """Check estimators on multi-output problems."""
    X = [[-2, -1],
         [-1, -1],
         [-1, -2],
         [1, 1],
         [1, 2],
         [2, 1],
         [-2, 1],
         [-1, 1],
         [-1, 2],
         [2, -1],
         [1, -1],
         [1, -2]]

    y = [[-1, 0],
         [-1, 0],
         [-1, 0],
         [1, 1],
         [1, 1],
         [1, 1],
         [-1, 2],
         [-1, 2],
         [-1, 2],
         [1, 3],
         [1, 3],
         [1, 3]]

    T = [[-1, -1], [1, 1], [-1, 1], [1, -1]]
    y_true = [[-1, 0], [1, 1], [-1, 2], [1, 3]]

    # toy classification problem
    for name, TreeClassifier in CLF_TREES.items():
        clf = TreeClassifier(random_state=0)
        y_hat = clf.fit(X, y).predict(T)
        assert_array_equal(y_hat, y_true)
        assert_equal(y_hat.shape, (4, 2))

        proba = clf.predict_proba(T)
        assert_equal(len(proba), 2)
        assert_equal(proba[0].shape, (4, 2))
        assert_equal(proba[1].shape, (4, 4))

        log_proba = clf.predict_log_proba(T)
        assert_equal(len(log_proba), 2)
        assert_equal(log_proba[0].shape, (4, 2))
        assert_equal(log_proba[1].shape, (4, 4))

    # toy regression problem
    for name, TreeRegressor in REG_TREES.items():
        reg = TreeRegressor(random_state=0)
        y_hat = reg.fit(X, y).predict(T)
        assert_almost_equal(y_hat, y_true)
        assert_equal(y_hat.shape, (4, 2))


def test_classes_shape():
    """Test that n_classes_ and classes_ have proper shape."""
    for name, TreeClassifier in CLF_TREES.items():
        # Classification, single output
        clf = TreeClassifier(random_state=0)
        clf.fit(X, y)

        assert_equal(clf.n_classes_, 2)
        assert_array_equal(clf.classes_, [-1, 1])

        # Classification, multi-output
        _y = np.vstack((y, np.array(y) * 2)).T
        clf = TreeClassifier(random_state=0)
        clf.fit(X, _y)
        assert_equal(len(clf.n_classes_), 2)
        assert_equal(len(clf.classes_), 2)
        assert_array_equal(clf.n_classes_, [2, 2])
        assert_array_equal(clf.classes_, [[-1, 1], [-2, 2]])


def test_unbalanced_iris():
    """Check class rebalancing."""
    unbalanced_X = iris.data[:125]
    unbalanced_y = iris.target[:125]
    sample_weight = balance_weights(unbalanced_y)

    for name, TreeClassifier in CLF_TREES.items():
        clf = TreeClassifier(random_state=0)
        clf.fit(unbalanced_X, unbalanced_y, sample_weight=sample_weight)
        assert_almost_equal(clf.predict(unbalanced_X), unbalanced_y)


def test_memory_layout():
    """Check that it works no matter the memory layout"""
    for (name, TreeEstimator), dtype in product(ALL_TREES.items(),
                                                [np.float64, np.float32]):
        est = TreeEstimator(random_state=0)

        # Nothing
        X = np.asarray(iris.data, dtype=dtype)
        y = iris.target
        assert_array_equal(est.fit(X, y).predict(X), y)

        # C-order
        X = np.asarray(iris.data, order="C", dtype=dtype)
        y = iris.target
        assert_array_equal(est.fit(X, y).predict(X), y)

        # F-order
        X = np.asarray(iris.data, order="F", dtype=dtype)
        y = iris.target
        assert_array_equal(est.fit(X, y).predict(X), y)

        # Contiguous
        X = np.ascontiguousarray(iris.data, dtype=dtype)
        y = iris.target
        assert_array_equal(est.fit(X, y).predict(X), y)

        # Strided
        X = np.asarray(iris.data[::3], dtype=dtype)
        y = iris.target[::3]
        assert_array_equal(est.fit(X, y).predict(X), y)


def test_sample_weight():
    """Check sample weighting."""
    # Test that zero-weighted samples are not taken into account
    X = np.arange(100)[:, np.newaxis]
    y = np.ones(100)
    y[:50] = 0.0

    sample_weight = np.ones(100)
    sample_weight[y == 0] = 0.0

    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X, y, sample_weight=sample_weight)
    assert_array_equal(clf.predict(X), np.ones(100))

    # Test that low weighted samples are not taken into account at low depth
    X = np.arange(200)[:, np.newaxis]
    y = np.zeros(200)
    y[50:100] = 1
    y[100:200] = 2
    X[100:200, 0] = 200

    sample_weight = np.ones(200)

    sample_weight[y == 2] = .51  # Samples of class '2' are still weightier
    clf = DecisionTreeClassifier(max_depth=1, random_state=0)
    clf.fit(X, y, sample_weight=sample_weight)
    assert_equal(clf.tree_.threshold[0], 149.5)

    sample_weight[y == 2] = .50  # Samples of class '2' are no longer weightier
    clf = DecisionTreeClassifier(max_depth=1, random_state=0)
    clf.fit(X, y, sample_weight=sample_weight)
    assert_equal(clf.tree_.threshold[0], 49.5)  # Threshold should have moved

    # Test that sample weighting is the same as having duplicates
    X = iris.data
    y = iris.target

    duplicates = rng.randint(0, X.shape[0], 200)

    clf = DecisionTreeClassifier(random_state=1)
    clf.fit(X[duplicates], y[duplicates])

    sample_weight = bincount(duplicates, minlength=X.shape[0])
    clf2 = DecisionTreeClassifier(random_state=1)
    clf2.fit(X, y, sample_weight=sample_weight)

    internal = clf.tree_.children_left != tree._tree.TREE_LEAF
    assert_array_almost_equal(clf.tree_.threshold[internal],
                              clf2.tree_.threshold[internal])


def test_32bit_equality():
    """Check if 32bit and 64bit get the same result. """
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target,
                                                    random_state=1)
    est = DecisionTreeRegressor(random_state=1)

    est.fit(X_train, y_train)
    score = est.score(X_test, y_test)
    assert_almost_equal(0.76624433012786, score)
