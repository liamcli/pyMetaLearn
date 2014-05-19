from collections import defaultdict
from collections import OrderedDict
import numpy as np
import sys

import scipy.stats
from scipy.linalg import LinAlgError

import sklearn
import sklearn.metrics
import sklearn.cross_validation

import scipy.stats

import time


class MetafeatureFunctions:
    def __init__(self):
        self.functions = OrderedDict()

    def __iter__(self):
        return self.functions.__iter__()

    def __getitem__(self, item):
        return self.functions.__getitem__(item)

    def __setitem__(self, key, value):
        return self.functions.__setitem__(key, value)

    def __delitem__(self, key):
        return self.functions.__delitem__(key)

    def __contains__(self, item):
        return self.functions.__contains__(item)

    def define(self, name):
        """Decorator for adding metafeature functions to a "dictionary" of
        metafeatures. This behaves like a function decorating a function,
        not a class decorating a function"""
        def wrapper(func):
            self.__setitem__(name, func)
            return func
        return wrapper
    
metafeatures = MetafeatureFunctions()


################################################################################
### Simple features
@metafeatures.define("number_of_instances")
def number_of_instances(X, Y):
    return float(X.shape[0])

@metafeatures.define("log_number_of_instances")
def log_number_of_instances(X, Y):
    return np.log(metafeatures["number_of_instances"](X, Y))

@metafeatures.define("number_of_classes")
def number_of_classes(X, Y):
    return float(len(np.unique(Y)))

@metafeatures.define("number_of_features")
def number_of_features(X, Y):
    return float(X.shape[1])

@metafeatures.define("log_number_of_features")
def number_of_features(X, Y):
    return np.log(metafeatures["number_of_features"](X, Y))

@metafeatures.define("number_of_Instances_with_missing_values")
def number_of_Instances_with_missing_values(X, Y):
    num_instances_with_missing_values = 0.
    for row in X.iterrows():
        if sum(row[1].isnull()) > 0:
            num_instances_with_missing_values += 1
    return num_instances_with_missing_values

@metafeatures.define("percentage_of_Instances_with_missing_values")
def percentage_of_Instances_with_missing_values(X, Y):
    return float(metafeatures["number_of_Instances_with_missing_values"](X, Y))\
           / float(metafeatures["number_of_instances"](X, Y))

@metafeatures.define("number_of_features_with_missing_values")
def number_of_features_with_missing_values(X, Y):
    num_features_with_missing_values = 0.
    for column in X.iteritems():
        if sum(column[1].isnull()) > 0:
            num_features_with_missing_values += 1
    return num_features_with_missing_values

@metafeatures.define("percentage_of_features_with_missing_values")
def percentage_of_features_with_missing_values(X, Y):
    return float(metafeatures["number_of_features_with_missing_values"](X, Y))\
           / float(metafeatures["number_of_features"](X, Y))

@metafeatures.define("number_of_missing_values")
def number_of_missing_values(X, Y):
    num_missing_values = 0.
    for column in X.iteritems():
        num_missing_values += sum(column[1].isnull())
    return num_missing_values

@metafeatures.define("percentage_of_missing_values")
def percentage_of_missing_values(X, Y):
    return float(metafeatures["number_of_missing_values"](X, Y)) / \
           float(X.shape[0]*X.shape[1])

@metafeatures.define("number_of_numeric_features")
def number_of_numeric_features(X, Y):
    num_numeric = 0.
    for column in X.iteritems():
        if column[1].dtype == np.float64:
            num_numeric += 1
    return num_numeric

@metafeatures.define("number_of_categorical_features")
def number_of_categorical_features(X, Y):
    num_categorical = 0.
    for column in X.iteritems():
        if column[1].dtype == 'object':
            num_categorical += 1
    return num_categorical

@metafeatures.define("ratio_numerical_to_categorical")
def ratio_numerical_to_nominal(X, Y):
    num_categorical = float(metafeatures["number_of_categorical_features"](X, Y))
    num_numerical = float(metafeatures["number_of_numeric_features"](X, Y))
    if num_categorical == 0.0:
        return 0.
    return num_numerical / num_categorical

@metafeatures.define("ratio_categorical_to_numerical")
def ratio_nominal_to_numerical(X, Y):
    num_categorical = float(metafeatures["number_of_categorical_features"](X, Y))
    num_numerical = float(metafeatures["number_of_numeric_features"](X, Y))
    if num_numerical == 0.0:
        return 0.
    else:
        return num_categorical / num_numerical

# Number of attributes divided by number of samples
@metafeatures.define("dataset_ratio")
def dataset_ratio(X, Y):
    return float(metafeatures["number_of_features"](X, Y)) /\
        float(metafeatures["number_of_instances"](X, Y))

@metafeatures.define("log_dataset_ratio")
def log_dataset_ratio(X, Y):
    return np.log(metafeatures["dataset_ratio"](X, Y))

@metafeatures.define("inverse_dataset_ratio")
def inverse_dataset_ratio(X, Y):
    return float(metafeatures["number_of_instances"](X, Y)) /\
        float(metafeatures["number_of_features"](X, Y))

@metafeatures.define("log_inverse_dataset_ratio")
def log_inverse_dataset_ratio(X, Y):
    return np.log(metafeatures["inverse_dataset_ratio"](X, Y))

@metafeatures.define("class_probability_min")
def class_probability_min(X, Y):
    occurence_dict = defaultdict(float)
    for value in Y.values:
        occurence_dict[value] += 1
    min_value = np.iinfo(np.int64).max
    for num_occurences in occurence_dict.itervalues():
        if num_occurences < min_value:
            min_value = num_occurences
    return float(min_value) / float(Y.size)

# aka default accuracy
@metafeatures.define("class_probability_max")
def class_probability_max(X, Y):
    occurence_dict = defaultdict(float)
    for value in Y.values:
        occurence_dict[value] += 1
    max_value = -1
    for num_occurences in occurence_dict.itervalues():
        if num_occurences > max_value:
            max_value = num_occurences
    return float(max_value) / float(Y.size)

@metafeatures.define("class_probability_mean")
def class_probability_mean(X, Y):
    occurence_dict = defaultdict(float)
    for value in Y.values:
        occurence_dict[value] += 1
    occurences = np.array([occurrence for occurrence in occurence_dict.itervalues()],
                         dtype=np.float64)
    return (occurences / Y.size).mean()

@metafeatures.define("class_probability_std")
def class_probability_std(X, Y):
    occurence_dict = defaultdict(float)
    for value in Y.values:
        occurence_dict[value] += 1
    occurences = np.array([occurrence for occurrence in occurence_dict.itervalues()],
                         dtype=np.float64)
    return (occurences / Y.size).std()

################################################################################
# Reif, A Comprehensive Dataset for Evaluating Approaches of various Meta-Learning Tasks
# defines these five metafeatures as simple metafeatures, but they could also
#  be the counterpart for the skewness and kurtosis of the numerical features
@metafeatures.define("symbols_min")
def symbols_min(X, Y):
    # The minimum can only be zero if there are no nominal features,
    # otherwise it is at least one
    # TODO: shouldn't this rather be two?
    minimum = sys.maxint
    for column in X.iteritems():
        if column[1].dtype == 'object':
            unique = column[1].nunique()
            if unique > 0 and unique < minimum:
                minimum = unique
    return minimum if minimum < sys.maxint else 0

@metafeatures.define("symbols_max")
def symbols_max(X, Y):
    maximum = 0
    for column in X.iteritems():
        if column[1].dtype == 'object':
            unique = column[1].nunique()
            maximum = max(unique, maximum)
    return maximum

@metafeatures.define("symbols_mean")
def symbols_mean(X, Y):
    uniques = []
    for column in X.iteritems():
        if column[1].dtype == 'object':
            unique = column[1].nunique()
            if unique > 0:
                uniques.append(unique)
    mean = np.nanmean(uniques)
    return mean if np.isfinite(mean) else 0

@metafeatures.define("symbols_std")
def symbols_std(X, Y):
    uniques = []
    for column in X.iteritems():
        if column[1].dtype == 'object':
            unique = column[1].nunique()
            if unique > 0:
                uniques.append(unique)
    std = np.nanstd(uniques)
    return std if np.isfinite(std) else 0

@metafeatures.define("symbols_sum")
def symbols_sum(X, Y):
    uniques = []
    for column in X.iteritems():
        if column[1].dtype == 'object':
            unique = column[1].nunique()
            if unique > 0:
                uniques.append(unique)
    sum = np.nansum(uniques)
    return sum if np.isfinite(sum) else 0

################################################################################
# Statistical meta features
# Only use third and fourth statistical moment because it is common to
# standardize for the other two
# see Engels & Theusinger, 1998 - Using a Data Metric for Preprocessing Advice for Data Mining Applications.

@metafeatures.define("kurtosis_min")
def kurtosis_min(X, Y):
    # The minimum can only be zero if there are no nominal features,
    # otherwise it is at least one
    # TODO: shouldn't this rather be two?
    kurts = []
    for column in X.iteritems():
        if column[1].dtype == np.float64:
            kurts.append(scipy.stats.kurtosis(column[1].values))
            # kurts.append(column[1].kurt())
    minimum = np.nanmin(kurts) if len(kurts) > 0 else 0
    return minimum if np.isfinite(minimum) else 0

@metafeatures.define("kurtosis_max")
def kurtosis_max(X, Y):
    kurts = []
    for column in X.iteritems():
        if column[1].dtype == np.float64:
            kurts.append(scipy.stats.kurtosis(column[1].values))
            # kurts.append(column[1].kurt())
    maximum = np.nanmax(kurts) if len(kurts) > 0 else 0
    return maximum if np.isfinite(maximum) else 0

@metafeatures.define("kurtosis_mean")
def kurtosis_mean(X, Y):
    kurts = []
    for column in X.iteritems():
        if column[1].dtype == np.float64:
            kurts.append(scipy.stats.kurtosis(column[1].values))
            # .append(column[1].kurt())
    mean = np.nanmean(kurts) if len(kurts) > 0 else 0
    return mean if np.isfinite(mean) else 0

@metafeatures.define("kurtosis_std")
def kurtosis_std(X, Y):
    kurts = []
    for column in X.iteritems():
        if column[1].dtype == np.float64:
            kurts.append(scipy.stats.kurtosis(column[1].values))
            # kurts.append(column[1].kurt())
    std = np.nanstd(kurts) if len(kurts) > 0 else 0
    return std if np.isfinite(std) else 0

@metafeatures.define("skewness_min")
def skewness_min(X, Y):
    # The minimum can only be zero if there are no nominal features,
    # otherwise it is at least one
    # TODO: shouldn't this rather be two?
    skews = []
    for column in X.iteritems():
        if column[1].dtype == np.float64:
            skews.append(scipy.stats.skew(column[1].values))
            # skews.append(column[1].skew())
    minimum = np.nanmin(skews) if len(skews) > 0 else 0
    return minimum if np.isfinite(minimum) else 0

@metafeatures.define("skewness_max")
def skewness_max(X, Y):
    skews = []
    for column in X.iteritems():
        if column[1].dtype == np.float64:
            skews.append(scipy.stats.skew(column[1].values))
            # skews.append(column[1].skew())
    maximum = np.nanmax(skews) if len(skews) > 0 else 0
    return maximum if np.isfinite(maximum) else 0

@metafeatures.define("skewness_mean")
def skewness_mean(X, Y):
    skews = []
    for column in X.iteritems():
        if column[1].dtype == np.float64:
            skews.append(scipy.stats.skew(column[1].values))
            #skews.append(column[1].skew())
    mean = np.nanmean(skews) if len(skews) > 0 else 0
    return mean if np.isfinite(mean) else 0

@metafeatures.define("skewness_std")
def skewness_std(X, Y):
    skews = []
    for column in X.iteritems():
        if column[1].dtype == np.float64:
            skews.append(scipy.stats.skew(column[1].values))
            #skews.append(column[1].skew())
    std = np.nanstd(skews) if len(skews) > 0 else 0
    return std if np.isfinite(std) else 0

#@metafeatures.define("cancor1")
#def cancor1(X, Y):
#    pass

#@metafeatures.define("cancor2")
#def cancor2(X, Y):
#    pass

################################################################################
# Information-theoretic metafeatures
@metafeatures.define("class_entropy")
def class_entroy(X, Y):
    occurence_dict = defaultdict(float)
    for value in Y.values:
        occurence_dict[value] += 1
    return scipy.stats.entropy([occurence_dict[key] for key in
                                occurence_dict], base=2)

#@metafeatures.define("normalized_class_entropy")

#@metafeatures.define("attribute_entropy")

#@metafeatures.define("normalized_attribute_entropy")

#@metafeatures.define("joint_entropy")

#@metafeatures.define("mutual_information")

#@metafeatures.define("noise-signal-ratio")

#@metafeatures.define("signal-noise-ratio")

#@metafeatures.define("equivalent_number_of_attributes")

#@metafeatures.define("conditional_entropy")

#@metafeatures.define("average_attribute_entropy")

################################################################################
# Landmarking features, computed with cross validation
# These should be invoked with the same transformations of X and Y with which
# sklearn will be called later on

# from Pfahringer 2000
# Linear discriminant learner
@metafeatures.define("landmark_lda")
def landmark_lda(X, Y):
    import sklearn.lda
    kf = sklearn.cross_validation.StratifiedKFold(Y, n_folds=10, indices=True)
    accuracy = 0.
    try:
        for train, test in kf:
            lda = sklearn.lda.LDA()
            lda.fit(X[train], Y[train])
            predictions = lda.predict(X[test])
            accuracy += sklearn.metrics.accuracy_score(predictions, Y[test])
        return 1 - (accuracy / 10)
    except scipy.linalg.LinAlgError as e:
        print "!!!", e, "Returned 1 instead!"
        return 1
    except ValueError as e:
        print "!!!", e, "Returned 1 instead"
        return 1

# Naive Bayes
@metafeatures.define("landmark_naive_bayes")
def landmark_naive_bayes(X, Y):
    import sklearn.naive_bayes
    kf = sklearn.cross_validation.StratifiedKFold(Y, n_folds=10, indices=True)
    accuracy = 0.
    for train, test in kf:
        nb = sklearn.naive_bayes.MultinomialNB()
        try:
            nb.fit(X[train], Y[train])
        except ValueError:
            print X[train]
            raise Exception
        predictions = nb.predict(X[test])
        accuracy += sklearn.metrics.accuracy_score(predictions, Y[test])
    return 1 - (accuracy / 10)

# Cart learner instead of C5.0
@metafeatures.define("landmark_decision_tree")
def landmark_decision_tree(X, Y):
    import sklearn.tree
    kf = sklearn.cross_validation.StratifiedKFold(Y, n_folds=10, indices=True)
    accuracy = 0.
    for train, test in kf:
        random_state = sklearn.utils.check_random_state(42)
        tree = sklearn.tree.DecisionTreeClassifier(random_state=random_state)
        tree.fit(X[train], Y[train])
        predictions = tree.predict(X[test])
        accuracy += sklearn.metrics.accuracy_score(predictions, Y[test])
    return 1 - (accuracy / 10)

"""If there is a dataset which has OneHotEncoded features it can happend that
a node learner splits at one of the attribute encodings. This should be fine
as the dataset is later on used encoded."""

# TODO: use the same forest, this has then to be computed only once and hence
#  saves a lot of time...
@metafeatures.define("landmark_decision_node_learner")
def landmark_decision_node_learner(X, Y):
    import sklearn.tree
    kf = sklearn.cross_validation.StratifiedKFold(Y, n_folds=10, indices=True)
    accuracy = 0.
    for train, test in kf:
        random_state = sklearn.utils.check_random_state(42)
        node = sklearn.tree.DecisionTreeClassifier(criterion="entropy",
            max_features=None, max_depth=1, min_samples_split=1,
            min_samples_leaf=1, random_state=random_state)
        node.fit(X[train], Y[train])
        predictions = node.predict(X[test])
        accuracy += sklearn.metrics.accuracy_score(predictions, Y[test])
    return 1 - (accuracy / 10)

@metafeatures.define("landmark_random_node_learner")
def landmark_random_node_learner(X, Y):
    import sklearn.tree
    kf = sklearn.cross_validation.StratifiedKFold(Y, n_folds=10, indices=True)
    accuracy = 0.
    rs = np.random.RandomState(int(42./np.log(X.shape[1])))
    attribute_idx = rs.randint(0, X.shape[1])

    for train, test in kf:
        random_state = sklearn.utils.check_random_state(42)
        node = sklearn.tree.DecisionTreeClassifier(
            criterion="entropy", max_depth=1, random_state=random_state,
            min_samples_split=1, min_samples_leaf=1, max_features=None)
        node.fit(X[train][:,attribute_idx].reshape((-1, 1)), Y[train])
        predictions = node.predict(X[test][:,attribute_idx].reshape((-1, 1)))
        accuracy += sklearn.metrics.accuracy_score(predictions, Y[test])
    return 1 - (accuracy / 10)

"""
This is wrong...
@metafeatures.define("landmark_worst_node_learner")
def landmark_worst_node_learner(X, Y):
    # TODO: this takes more than 10 minutes on some datasets (eg mfeat-pixels)
    # which has 240*6 = 1440 discrete attributes...
    # TODO: calculate information gain instead of using the worst test result
    import sklearn.tree
    performances = []
    for attribute_idx in range(X.shape[1]):
        kf = sklearn.cross_validation.StratifiedKFold(Y, n_folds=10, indices=True)
        accuracy = 0.
        for train, test in kf:
            node = sklearn.tree.DecisionTreeClassifier(criterion="entropy",
                max_features=None, max_depth=1, min_samples_split=1,
                min_samples_leaf=1)
            node.fit(X[train][:,attribute_idx].reshape((-1, 1)), Y[train],
                     check_input=False)
            predictions = node.predict(X[test][:,attribute_idx].reshape((-1, 1)))
            accuracy += sklearn.metrics.accuracy_score(predictions, Y[test])
        performances.append(1 - (accuracy / 10))
    return max(performances)
"""

# Replace the Elite 1NN with a normal 1NN, this slightly changes the
# intuition behind this landmark, but Elite 1NN is used nowhere else...
@metafeatures.define("landmark_1NN")
def landmark_1NN(X, Y):
    import sklearn.neighbors
    kf = sklearn.cross_validation.StratifiedKFold(Y, n_folds=10, indices=True)
    accuracy = 0.
    for train, test in kf:
        lda = sklearn.neighbors.KNeighborsClassifier(1)
        lda.fit(X[train], Y[train])
        predictions = lda.predict(X[test])
        accuracy += sklearn.metrics.accuracy_score(predictions, Y[test])
    return 1 - (accuracy / 10)

################################################################################
# Bardenet 2013 - Collaborative Hyperparameter Tuning
# K number of classes ("number_of_classes")
# log(d), log(number of attributes)
# log(n/d), log(number of training instances/number of attributes)
# p, how many principal components to keep in order to retain 95% of the
#     dataset variance
# skewness of a dataset projected onto one principal component...
# kurtosis of a dataset projected onto one principal component

pca_object = None

# Maybe define some more...
@metafeatures.define("pca_95%")
def pca_95percent(X, Y):
    if pca_object is None:
        import sklearn.decomposition
        pca = sklearn.decomposition.PCA(copy=True)
        rs = np.random.RandomState(42)
        indices = np.arange(X.shape[0])
        for i in range(10):
            try:
                rs.shuffle(indices)
                pca.fit(X)
                break
            except LinAlgError as e:
                pass
            print "Wasn't able to fit a PCA."
            return 1    # Needs all components...

        global pca_object
        pca_object = pca
    else:
        pca = pca_object

    sum_ = 0.
    idx = 0
    while sum_ < 0.95:
        sum_ += pca.explained_variance_ratio_[idx]
        idx += 1
    return float(idx)/float(X.shape[1])

# Kurtosis of first PC
@metafeatures.define("pca_kurtosis_first_pc")
def pca_kurtosis_first_pc(X, Y):
    if pca_object is None:
        import sklearn.decomposition
        pca = sklearn.decomposition.PCA(copy=True)
        rs = np.random.RandomState(42)
        indices = np.arange(X.shape[0])
        for i in range(10):
            try:
                rs.shuffle(indices)
                pca.fit(X)
                break
            except LinAlgError as e:
                pass
            print "Wasn't able to fit a PCA."
            return 0    # Some default...

        global pca_object
        pca_object = pca
    else:
        pca = pca_object

    components = pca.components_
    pca.components_ = components[:1]
    transformed = pca.transform(X)
    pca.components_ = components

    kurtosis = scipy.stats.kurtosis(transformed)
    return kurtosis[0]

# Skewness of first PC
@metafeatures.define("pca_skewness_first_pc")
def pca_skewness_first_pc(X, Y):
    if pca_object is None:
        import sklearn.decomposition
        pca = sklearn.decomposition.PCA(copy=True)
        rs = np.random.RandomState(42)
        indices = np.arange(X.shape[0])
        for i in range(10):
            try:
                rs.shuffle(indices)
                pca.fit(X)
                break
            except LinAlgError as e:
                pass
            print "Wasn't able to fit a PCA."
            return 0    # Some default...

        global pca_object
        pca_object = pca
    else:
        pca = pca_object

    components = pca.components_
    pca.components_ = components[:1]
    transformed = pca.transform(X)
    pca.components_ = components

    skewness = scipy.stats.skew(transformed)
    return skewness[0]

def calculate_all_metafeatures(dataset, subset_indices=None,
        calculate=None, dont_calculate=None):
    mf = OrderedDict()
    Xnpy = Ynpy = Xpd = Ypd = None

    for name in metafeatures:
        if calculate is not None and name not in calculate:
            continue
        if dont_calculate is not None and name in dont_calculate:
            continue

        if Xnpy is None:
            Xnpy, Ynpy = dataset.get_npy(scaling="scale")
            Xpd, Ypd = dataset.get_pandas()

            if subset_indices is not None:
                subset_indices = np.array([True if i in subset_indices else False for \
                        i in range(Xnpy.shape[0])])
                Xnpy = Xnpy[subset_indices]
                Ynpy = Ynpy[subset_indices]
                Xpd = Xpd[subset_indices]
                Ypd = Ypd[subset_indices]

        if name in npy_metafeatures:
            # This is not only important for datasets which are somehow
            # sorted in a strange way, but also provents lda from failing in
            # some cases...
            X, Y = Xnpy, Ynpy
            rs = np.random.RandomState(42)
            indices = np.arange(X.shape[0])
            rs.shuffle(indices)
            X = X[indices]
            Y = Y[indices]
        else:
            X, Y = Xpd, Ypd

        starttime = time.time()
        try:
            mf[name] = metafeatures[name](X, Y)
            print name, "took", time.time() - starttime, "seconds, result", \
                 mf[name]
        except Exception as e:
            print "!!!", name, "Failed", e


    # Delete intermediate variables
    global pca_object
    pca_object = None
    return mf


npy_metafeatures = set(["landmark_lda", "landmark_naive_bayes",
                        "landmark_decision_tree",
                        "landmark_decision_node_learner",
                        "landmark_random_node_learner",
                        "landmark_worst_node_learner", "landmark_1NN",
                        "pca_95%", "pca_kurtosis_first_pc", "pca_skewness_first_pc"])

subsets = dict()
# All implemented metafeatures
subsets["all"] = set(metafeatures.functions.keys())

# Metafeatures used by Pfahringer et al. (2000) in the first experiment
subsets["pfahringer_2000_experiment1"] = set(["number_of_features",
                                             "number_of_numeric_features",
                                             "number_of_categorical_features",
                                             "number_of_classes",
                                             "class_probability_max",
                                             "landmark_lda",
                                             "landmark_naive_bayes",
                                             "landmark_decision_tree"])

# Metafeatures used by Pfahringer et al. (2000) in the second experiment
# worst node learner not implemented yet
"""
pfahringer_2000_experiment2 = set(["landmark_decision_node_learner",
                                   "landmark_random_node_learner",
                                   "landmark_worst_node_learner",
                                   "landmark_1NN"])
"""

# Metafeatures used by Yogotama and Mann (2014)
subsets["yogotama_2014"] = set(["log_number_of_features",
                               "log_number_of_instances",
                               "number_of_classes"])

# Metafeatures used by Bardenet et al. (2013) for the AdaBoost.MH experiment
subsets["bardenet_2013_boost"] = set(["number_of_classes",
                                     "log_number_of_features",
                                     "log_inverse_dataset_ratio", "pca_95%"])

# Metafeatures used by Bardenet et al. (2013) for the Neural Net experiment
subsets["bardenet_2013_nn"] = set(["number_of_classes",
                                  "log_number_of_features",
                                  "log_inverse_dataset_ratio",
                                  "pca_kurtosis_first_pc",
                                  "pca_skewness_first_pc"])


