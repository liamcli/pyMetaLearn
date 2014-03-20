from collections import defaultdict
from collections import OrderedDict
import numpy as np


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

@metafeatures.define("number_of_classes")
def number_of_classes(X, Y):
    return float(len(np.unique(Y)))

@metafeatures.define("number_of_features")
def number_of_features(X, Y):
    return float(X.shape[1])

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
@metafeatures.define("dimensionality")
def dimensionality(X, Y):
    return float(metafeatures["number_of_features"](X, Y)) /\
        float(metafeatures["number_of_instances"](X, Y))

@metafeatures.define("inverse_dimensionality")
def inverse_dimensionality(X, Y):
    return float(metafeatures["number_of_instances"](X, Y)) /\
        float(metafeatures["number_of_features"](X, Y))

"""
@metafeatures.define("nominal_min(")
def nominal_min(X, Y):
    pass

@metafeatures.define("nominal_max")
def nominal_max(X, Y):
    pass

@metafeatures.define("nominal_mean")
def nominal_mean(X, Y):
    pass

@metafeatures.define("nominal_std")
def nominal_std(X, Y):
    pass

@metafeatures.define("nominal_sum")
def nominal_sum(X, Y):
    pass
"""

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

def calculate_all_metafeatures(X, Y):
    mf = OrderedDict()
    for name in metafeatures:
        mf[name] = metafeatures[name](X, Y)
    return mf

def calculate_missing_metafeatures(X, Y, dict_):
    raise NotImplementedError()