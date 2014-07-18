from collections import defaultdict, OrderedDict, deque
import logging
import numpy as np
import sys

import scipy.stats
from scipy.linalg import LinAlgError

import sklearn
import sklearn.metrics
import sklearn.cross_validation

import scipy.stats

import time


class HelperFunctions:
    def __init__(self):
        self.functions = OrderedDict()
        self.values = OrderedDict()
        self.computation_time = OrderedDict()

    def clear(self):
        self.values = OrderedDict()
        self.computation_time = OrderedDict()

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

    def is_calculated(self, key):
        """Return if a helper function has already been executed.

        Necessary as get_value() can return None if the helper function hasn't
        been executed or if it returned None."""
        return key in self.values

    def get_value(self, key):
        return self.values.get(key)

    def get_computation_time(self, key):
        return self.computation_time[key]

    def set_value(self, key, item, time_):
        self.values[key] = item
        self.computation_time[key] = time_

    def define(self, name):
        """Decorator for adding helper functions to a "dictionary".
        This behaves like a function decorating a function,
        not a class decorating a function"""
        def wrapper(func):
            self.__setitem__(name, func)
            return func
        return wrapper


class MetafeatureFunctions:
    def __init__(self):
        self.functions = OrderedDict()
        self.dependencies = OrderedDict()
        self.values = OrderedDict()
        self.computation_time = OrderedDict()

    def clear(self):
        self.values = OrderedDict()
        self.computation_time = OrderedDict()

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

    def get_value(self, key):
        return self.values[key]

    def get_computation_time(self, key):
        return self.computation_time[key]

    def set_value(self, key, item, time_):
        self.values[key] = item
        self.computation_time[key] = time_

    def is_calculated(self, key):
        """Return if a helper function has already been executed.

        Necessary as get_value() can return None if the helper function hasn't
        been executed or if it returned None."""
        return key in self.values

    def get_dependency(self, name):
        """Return the dependency of metafeature "name".
        """
        return self.dependencies.get(name)

    def define(self, name, dependency=None):
        """Decorator for adding metafeature functions to a "dictionary" of
        metafeatures. This behaves like a function decorating a function,
        not a class decorating a function"""
        def wrapper(func):
            self.__setitem__(name, func)
            self.dependencies[name] = dependency
            return func
        return wrapper
    
metafeatures = MetafeatureFunctions()
helper_functions = HelperFunctions()


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

@helper_functions.define("missing_values")
def missing_values(X, Y):
    missing = X.isnull()
    return missing

@metafeatures.define("number_of_Instances_with_missing_values",
                     dependency="missing_values")
def number_of_Instances_with_missing_values(X, Y):
    missing = helper_functions.get_value("missing_values")
    num_missing = missing.sum(axis=1)
    return float(sum([1 if num > 0 else 0 for num in num_missing]))

@metafeatures.define("percentage_of_Instances_with_missing_values",
                     dependency="number_of_Instances_with_missing_values")
def percentage_of_Instances_with_missing_values(X, Y):
    return float(metafeatures.get_value("number_of_Instances_with_missing_values")\
           / float(metafeatures["number_of_instances"](X, Y)))

@metafeatures.define("number_of_features_with_missing_values",
                     dependency="missing_values")
def number_of_features_with_missing_values(X, Y):
    missing = helper_functions.get_value("missing_values")
    num_missing = missing.sum(axis=0)
    return float(sum([1 if num > 0 else 0 for num in num_missing]))

@metafeatures.define("percentage_of_features_with_missing_values",
                     dependency="number_of_features_with_missing_values")
def percentage_of_features_with_missing_values(X, Y):
    return float(metafeatures.get_value("number_of_features_with_missing_values")\
           / float(metafeatures["number_of_features"](X, Y)))

@metafeatures.define("number_of_missing_values", dependency="missing_values")
def number_of_missing_values(X, Y):
    return float(sum(helper_functions.get_value("missing_values").sum()))

@metafeatures.define("percentage_of_missing_values",
                     dependency="number_of_missing_values")
def percentage_of_missing_values(X, Y):
    return float(metafeatures.get_value("number_of_missing_values")) / \
           float(X.shape[0]*X.shape[1])

# TODO: generalize this!
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

@helper_functions.define("class_occurences")
def class_occurences(X, Y):
    occurence_dict = defaultdict(float)
    for value in Y.values:
        occurence_dict[value] += 1
    return occurence_dict

@metafeatures.define("class_probability_min", dependency="class_occurences")
def class_probability_min(X, Y):
    occurence_dict = helper_functions.get_value("class_occurences")
    min_value = np.iinfo(np.int64).max
    for num_occurences in occurence_dict.itervalues():
        if num_occurences < min_value:
            min_value = num_occurences
    return float(min_value) / float(Y.size)

# aka default accuracy
@metafeatures.define("class_probability_max", dependency="class_occurences")
def class_probability_max(X, Y):
    occurence_dict = helper_functions.get_value("class_occurences")
    max_value = -1
    for num_occurences in occurence_dict.itervalues():
        if num_occurences > max_value:
            max_value = num_occurences
    return float(max_value) / float(Y.size)

@metafeatures.define("class_probability_mean", dependency="class_occurences")
def class_probability_mean(X, Y):
    occurence_dict = helper_functions.get_value("class_occurences")
    occurences = np.array([occurrence for occurrence in occurence_dict.itervalues()],
                         dtype=np.float64)
    return (occurences / Y.size).mean()

@metafeatures.define("class_probability_std", dependency="class_occurences")
def class_probability_std(X, Y):
    occurence_dict = helper_functions.get_value("class_occurences")
    occurences = np.array([occurrence for occurrence in occurence_dict.itervalues()],
                         dtype=np.float64)
    return (occurences / Y.size).std()

################################################################################
# Reif, A Comprehensive Dataset for Evaluating Approaches of various Meta-Learning Tasks
# defines these five metafeatures as simple metafeatures, but they could also
#  be the counterpart for the skewness and kurtosis of the numerical features
@helper_functions.define("num_symbols")
def num_symbols(X, Y):
    symbols_per_column = []
    for column in X.iteritems():
        if column[1].dtype == 'object':
            symbols_per_column.append(column[1].nunique())
    return symbols_per_column

@metafeatures.define("symbols_min", dependency="num_symbols")
def symbols_min(X, Y):
    # The minimum can only be zero if there are no nominal features,
    # otherwise it is at least one
    # TODO: shouldn't this rather be two?
    minimum = sys.maxint
    for unique in helper_functions.get_value("num_symbols"):
        if unique > 0 and unique < minimum:
            minimum = unique
    return minimum if minimum < sys.maxint else 0

@metafeatures.define("symbols_max", dependency="num_symbols")
def symbols_max(X, Y):
    values = helper_functions.get_value("num_symbols")
    if len(values) == 0:
        return 0
    return max(max(values), 0)

@metafeatures.define("symbols_mean")
def symbols_mean(X, Y):
    # TODO: categorical attributes without a symbol don't count towards this
    # measure
    values = [val for val in helper_functions.get_value("num_symbols") if val > 0]
    mean = np.nanmean(values)
    return mean if np.isfinite(mean) else 0

@metafeatures.define("symbols_std", dependency="num_symbols")
def symbols_std(X, Y):
    values = [val for val in helper_functions.get_value("num_symbols") if val > 0]
    std = np.nanstd(values)
    return std if np.isfinite(std) else 0

@metafeatures.define("symbols_sum", dependency="num_symbols")
def symbols_sum(X, Y):
    sum = np.nansum(helper_functions.get_value("num_symbols"))
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
        return accuracy / 10
    except scipy.linalg.LinAlgError as e:
        logging.warning("LDA failed: %s Returned 0 instead!" % e)
        return 0
    except ValueError as e:
        logging.warning("LDA failed: %s Returned 0 instead!" % e)
        return 0

# Naive Bayes
@metafeatures.define("landmark_naive_bayes")
def landmark_naive_bayes(X, Y):
    import sklearn.naive_bayes
    kf = sklearn.cross_validation.StratifiedKFold(Y, n_folds=10, indices=True)
    accuracy = 0.
    for train, test in kf:
        nb = sklearn.naive_bayes.GaussianNB()
        nb.fit(X[train], Y[train])
        predictions = nb.predict(X[test])
        accuracy += sklearn.metrics.accuracy_score(predictions, Y[test])
    return accuracy / 10

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
    return accuracy / 10

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
    return accuracy / 10

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
    return accuracy / 10

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
    return accuracy / 10

################################################################################
# Bardenet 2013 - Collaborative Hyperparameter Tuning
# K number of classes ("number_of_classes")
# log(d), log(number of attributes)
# log(n/d), log(number of training instances/number of attributes)
# p, how many principal components to keep in order to retain 95% of the
#     dataset variance
# skewness of a dataset projected onto one principal component...
# kurtosis of a dataset projected onto one principal component


@helper_functions.define("PCA")
def pca(X, Y):
    import sklearn.decomposition
    pca = sklearn.decomposition.PCA(copy=True)
    rs = np.random.RandomState(42)
    indices = np.arange(X.shape[0])
    for i in range(10):
        try:
            rs.shuffle(indices)
            pca.fit(X)
            return pca
        except LinAlgError as e:
            pass
    logging.warning("Failed to compute a Principle Component Analysis")
    return None

# Maybe define some more...
@metafeatures.define("pca_95percent", dependency="PCA")
def pca_95percent(X, Y):
    pca_ = helper_functions.get_value("PCA")
    if pca_ is None:
        return 1
    sum_ = 0.
    idx = 0
    while sum_ < 0.95:
        sum_ += pca_.explained_variance_ratio_[idx]
        idx += 1
    return float(idx)/float(X.shape[1])

# Kurtosis of first PC
@metafeatures.define("pca_kurtosis_first_pc", dependency="PCA")
def pca_kurtosis_first_pc(X, Y):
    pca_ = helper_functions.get_value("PCA")
    if pca_ is None:
        return 0
    components = pca_.components_
    pca_.components_ = components[:1]
    transformed = pca_.transform(X)
    pca_.components_ = components

    kurtosis = scipy.stats.kurtosis(transformed)
    return kurtosis[0]

# Skewness of first PC
@metafeatures.define("pca_skewness_first_pc", dependency="PCA")
def pca_skewness_first_pc(X, Y):
    pca_ = helper_functions.get_value("PCA")
    if pca_ is None:
        return 0
    components = pca_.components_
    pca_.components_ = components[:1]
    transformed = pca_.transform(X)
    pca_.components_ = components

    skewness = scipy.stats.skew(transformed)
    return skewness[0]

def calculate_all_metafeatures(dataset, subset_indices=None,
        calculate=None, dont_calculate=None, return_times=None):
    helper_functions.clear()
    metafeatures.clear()
    mf = OrderedDict()
    times = OrderedDict()
    Xnpy = Ynpy = Xpd = Ypd = None

    visited = set()
    to_visit = deque()
    to_visit.extend(metafeatures)
    while len(to_visit) > 0:
        dataset_just_loaded = False
        name = to_visit.pop()
        if calculate is not None and name not in calculate:
            continue
        if dont_calculate is not None and name in dont_calculate:
            continue

        # Lazily load the dataset only if its needed
        if Xnpy is None:
            Xnpy, Ynpy = dataset.get_npy(scaling="scale")
            Xpd, Ypd = dataset.get_pandas()
            dataset_just_loaded = True

        if dataset_just_loaded and subset_indices is not None:
            Xnpy = Xnpy[subset_indices]
            Ynpy = Ynpy[subset_indices]
            Xpd = Xpd.iloc[subset_indices]
            Ypd = Ypd.iloc[subset_indices]

        if dataset_just_loaded:
            # This is not only important for datasets which are somehow
            # sorted in a strange way, but also prevents lda from failing in
            # some cases...
            rs = np.random.RandomState(42)
            indices = np.arange(Xnpy.shape[0])
            rs.shuffle(indices)
            Xnpy = Xnpy[indices]
            Ynpy = Ynpy[indices]

        if name in npy_metafeatures:
            X, Y = Xnpy, Ynpy
        else:
            X, Y = Xpd, Ypd

        dependency = metafeatures.get_dependency(name)
        if dependency is not None:
            is_metafeature = dependency in metafeatures
            is_helper_function = dependency in helper_functions

            if is_metafeature and is_helper_function:
                raise NotImplementedError()
            elif not is_metafeature and not is_helper_function:
                raise ValueError(dependency)
            elif is_metafeature and not metafeatures.is_calculated(dependency):
                to_visit.appendleft(name)
                continue
            elif is_helper_function and not helper_functions.is_calculated(
                    dependency):
                logging.info("Starting to calculate dependency %s", name)
                dependency_start_time = time.time()
                value = helper_functions[dependency](X, Y)
                endtime = time.time()
                helper_functions.set_value(dependency, value,
                                       endtime - dependency_start_time)

                logging.info("Dependency %s took %f seconds.", dependency,
                    endtime - dependency_start_time)
                times[dependency] = endtime - dependency_start_time

        logging.info("Starting to calculate %s...", name)
        starttime = time.time()
        value = metafeatures[name](X, Y)
        endtime = time.time()
        metafeatures.set_value(name, value, endtime - starttime)
        logging.info("%s took %d seconds: %f", name, endtime - starttime,
                     metafeatures.get_value(name))
        mf[name] = value
        times[name] = endtime - starttime
        visited.add(name)

    if return_times:
        return mf, times
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


