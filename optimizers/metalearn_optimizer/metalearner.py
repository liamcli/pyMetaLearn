import argparse
import cPickle
from operator import itemgetter, attrgetter
from collections import namedtuple
from collections import OrderedDict
import logging
import functools
import os
import numpy as np
import sys

import sklearn.preprocessing as preprocessing
import pandas as pd

from pyMetaLearn.optimizers.metalearn_optimizer.meta_base import MetaBase
from pyMetaLearn.dataset_base import DatasetBase
from pyMetaLearn.openml.openml_task import OpenMLTask
import pyMetaLearn.optimizers.optimizer_base as optimizer_base
import pyMetaLearn.openml.manage_openml_data
import HPOlib.wrapping_util as wrapping_util

logging.basicConfig(format='[%(levelname)s] [%(asctime)s:%(name)s] %('
                           'message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("METALEARNING")
logger.setLevel(logging.INFO)

Experiment = namedtuple("Experiment", ["params", "result"])


def parse_parameters(args=None):
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-a", "--algorithm")
    group.add_argument("--cli_target")
    # parser.add_argument("-p", "--params", required=True)
    parser.add_argument("--cwd")
    parser.add_argument("--number_of_jobs", required=True, type=int)
    args = parser.parse_args(args=args)
    return args


def setup(args):
    context = dict()
    config = wrapping_util.load_experiment_config_file()
    test_fold = config.getint("EXPERIMENT", "test_fold")
    test_folds = config.getint("EXPERIMENT", "test_folds")
    context["distance_measure"] = config.get("METALEARNING", "distance_measure")
    context["subset_indices"] = dict()
    context["metafeatures_subset"] = config.get("METALEARNING",
                                                "metafeatures_subset")
    context["openml_data_dir"] = config.get("EXPERIMENT", "openml_data_dir")
    pyMetaLearn.openml.manage_openml_data.set_local_directory(context[
        "openml_data_dir"])
    # TODO: load the task for the dataset name...
    task_file = config.get("EXPERIMENT", "task_args_pkl")
    with open(task_file) as fh:
        task_args = cPickle.load(fh)
    task = OpenMLTask(**task_args)
    dataset = pyMetaLearn.openml.manage_openml_data.get_local_dataset(task.dataset_id)
    context["dataset_name"] = dataset._name
    context["task"] = task
    X, Y = task.get_dataset()
    train_splits, test_splits = task._get_fold(X, Y, fold=test_fold, folds=test_folds)
    train_splits = tuple(train_splits)
    context["subset_indices"][dataset._id] = train_splits
    base = DatasetBase()

    dataset_keys_file = config.get("METALEARNING", "datasets")
    with open(dataset_keys_file) as fh:
        dataset_keys = read_dataset_list(fh)

    datasets = list()
    for dataset_key in dataset_keys:
        logger.warning(str(dataset_key))
        datasets.append(base.get_dataset_from_key(dataset_key))

        task_file = os.path.join(pyMetaLearn
                .openml.manage_openml_data.get_local_directory(),
                "custom_tasks", "did_%d.pkl" % datasets[-1]._id)

        with open(task_file) as fh:
            task_args = cPickle.load(fh)
        task = OpenMLTask(**task_args)
        X, Y = task.get_dataset()
        train_splits, test_splits = task._get_fold(X, Y, fold=test_fold, folds=test_folds)
        train_splits = tuple(train_splits)
        context["subset_indices"][datasets[-1]._id] = train_splits

    experiments_list_file = config.get("METALEARNING", "experiments")
    with open(experiments_list_file) as fh:
        experiments_list = read_experiments_list(fh)

    experiments = []
    for dataset_experiments in experiments_list:
        experiments.append(list())
        for pickle_file in dataset_experiments:
            with open(pickle_file) as pkl:
                experiments[-1].extend(read_experiment_pickle(pkl))

    meta_base = MetaBase(datasets, experiments)
    context["meta_base"] = meta_base

    series = []
    subset_indices = context["subset_indices"]
    metafeature_subset = context["metafeatures_subset"]
    logger.info("%s", str(metafeature_subset))
    subset = pyMetaLearn.metafeatures.metafeatures.subsets[metafeature_subset]
    logger.info("%s", str(subset))
    for key in meta_base.get_datasets():
        did = meta_base.get_datasets()[key]._id
        series.append(meta_base.get_metadata_as_pandas(key,
            subset_indices=subset_indices[did],
            metafeature_subset=metafeature_subset))

    metafeatures = pd.DataFrame(series)
    context["metafeatures"] = metafeatures

    return context


def read_dataset_list(fh):
    dataset_filenames = list()
    for line  in fh:
        line = line.replace("\n", "")
        if line:
            dataset_filenames.append(line)
        else:
            raise ValueError("Empty lines in the dataset list are not "
                             "supported.")
    return dataset_filenames


def read_experiments_list(fh):
    experiments_list = list()
    for line in fh.readlines():
        experiments_list.append(line.split())
    return experiments_list


def read_experiment_pickle(fh):
    experiments = list()
    trials = cPickle.load(fh)
    for trial in trials["trials"]:
        experiments.append(Experiment(trial["params"], trial["result"]))
    return experiments


def l1(d1, d2):
    """http://en.wikipedia.org/wiki/Norm_%28mathematics%29#Taxicab_norm_or_Manhattan_norm"""
    return sum(abs(d1 - d2))


def l2(d1, d2):
    """http://en.wikipedia.org/wiki/Norm_%28mathematics%29#p-norm"""
    return np.sqrt(np.sum((d1 - d2)**2))


def vector_space_model(d1, d2):
    """http://en.wikipedia.org/wiki/Vector_space_model"""
    vector1 = preprocessing.normalize(np.array(d1.values, dtype=np.float64)
                                      .reshape((1, -1)), norm='l2')
    vector2 = preprocessing.normalize(np.array(d2.values, dtype=np.float64)
                                      .reshape((1, -1)), norm='l2')
    return np.sum(vector1 * vector2, axis=1)[0]


def calculate_distances(dataset_metafeatures, metafeatures, distance_fn):
    """Calculate distances between the dataset_metafeatures and all the
    dataset metafeatures from all datasets inside metafeatures.

    inputs:
    * dataset_metafeatures: A pd.Series for the source dataset
    * metafeatures: A pd.DataFrame with a row for every dataset for which the
        distance to source should be computed.
    * distance_fn: A callable which computes the distance

    output:
    * An array of tuples (distance, dataset_name) sorted by the distances
        (in ascending order)
    """
    distances = []
    assert isinstance(metafeatures, pd.DataFrame)
    assert metafeatures.values.dtype == np.float64
    assert np.isfinite(metafeatures.values).all()
    assert np.isfinite(dataset_metafeatures.values).all()

    for idx, candidate_metafeatures in metafeatures.iterrows():
        dist = distance_fn(dataset_metafeatures, candidate_metafeatures)
        dist_tuple = (dist, candidate_metafeatures.name)
        distances.append(dist_tuple)

    # Sort datasets according to distance to the target dataset
    distances.sort(key=itemgetter(0))
    return distances


def rescale(metafeatures):
    """Rescales all metafeatures inside a pd.DataFrame between 0 and 1"""
    assert isinstance(metafeatures, pd.DataFrame)
    assert metafeatures.values.dtype == np.float64
    # I also need to scale the target dataset meta features...
    mins = metafeatures.min()
    maxs = metafeatures.max()
    metafeatures = (metafeatures - mins) / (maxs - mins)
    return metafeatures


def find_best_hyperparams(experiments):
    min = sys.maxint
    idx = -1
    for i, value in enumerate(experiments):
        if value.result < min:
            min = value.result
            idx = i
    p = experiments[idx].params
    return p


def split_metafeature_array(dataset_name, metafeatures):
    """Split the metafeature array into dataset metafeatures and all other.

    This is inside an extra function for testing purpose.
    """
    dataset_metafeatures = metafeatures.ix[dataset_name].copy()
    metafeatures = metafeatures[metafeatures.index != dataset_name]
    return dataset_metafeatures, metafeatures


def assemble_best_hyperparameters_list(best_hyperparameters, distances):
    """Create an array with the best hyperparameters according to
    metalearning, sorted in ascending order, the nearest dataset first.
    Duplicates are removed"""
    hyperparameters = []
    sorted_names = []
    sorted_distances = []
    hyperparameters_set = set()
    for dist, name in distances:
        params_for_name = best_hyperparameters[name]
        if str(params_for_name) not in hyperparameters_set:
            hyperparameters.append(params_for_name)
            sorted_names.append(name)
            sorted_distances.append(dist)
            hyperparameters_set.add(str(params_for_name))

    return hyperparameters, sorted_names, sorted_distances


def select_params(best_hyperparameters, distances, history):
    # Iterate over all datasets which are sorted ascending by distance
    hyperparameters, sorted_names, sorted_distances = \
        assemble_best_hyperparameters_list(best_hyperparameters, distances)
    for idx, params in enumerate(hyperparameters):
        already_evaluated = False
        # Check if that dataset was already evaluated
        for experiment in history:
            # If so, return to the outer loop
            if params == experiment.params:
                already_evaluated = True
                break
        if not already_evaluated:
            logger.info("Nearest dataset with hyperparameters of best value "
                        "not evaluated yet is %s with a distance of %f" %
                        (sorted_names[idx], sorted_distances[idx]))
            return params
    raise StopIteration("No more values available.")


def metalearn_base(context):
    metafeatures = context["metafeatures"]
    meta_base = context["meta_base"]

    logger.info("Unscaled metafeatures %s", str(metafeatures))
    # Calculate the distance between the dataset and the other datasets
    assert metafeatures.values.dtype == np.float64
    # For l1 and l2 norm the metafeatures must be scaled between 0 and 1
    if "l1" in context["distance_measure"] or "l2" in context["distance_measure"]:
        metafeatures = rescale(metafeatures)

    dataset_metafeatures, metafeatures = split_metafeature_array(
        context["dataset_name"], metafeatures)
    distance_fn = getattr(sys.modules[__name__], context["distance_measure"])
    logger.info("Dataset Metafeatures %s" % str(dataset_metafeatures))
    logger.info("Metafeatures %s" % str(metafeatures))
    distances = calculate_distances(dataset_metafeatures, metafeatures,
                                    distance_fn)
    logger.info("Distances %s" % str(distances))

    best_hyperparameters = dict()
    for dataset in meta_base.get_datasets():
        experiments = meta_base.get_experiment(dataset)
        best_hyperparameters[dataset] = find_best_hyperparams(experiments)
    return best_hyperparameters, distances


def metalearn_suggest_all(param_space, context):
    best_hyperparameters, distances = metalearn_base(context)
    hp_list, name_list, dist_list = assemble_best_hyperparameters_list(
        best_hyperparameters, distances)
    for idx in range(len(hp_list)):
        logger.info("%s %s %s" % (hp_list[idx], name_list[idx], dist_list[idx]))
    return hp_list


def metalearn_suggest(history, param_space, context):
    best_hyperparameters, distances = metalearn_base(context)

    params_for_name = select_params(best_hyperparameters, distances, history)
    return params_for_name


def perform_optimization(target_algorithm, suggest_function, param_space,
                         context, time_budget=None, evaluation_budget=None):
    time_taken = 0
    num_evaluations = 0
    evaluations = []

    while True:
        if time_taken >= time_budget:
            logger.info("Reached time budget. Exiting optimization.")
            break
        if num_evaluations >= evaluation_budget:
            logger.info("Reached maximum number of evaluations. Exiting "
                        "optimization.")
            break

        params = suggest_function(evaluations, param_space, context)

        fixed_params = OrderedDict()
        # Hack to remove all trailing - from the params which are
        # accidently in the experiment pickle of the current HPOlib version
        for key in params:
            if key[0] == "-":
                fixed_params[key[1:]] = params[key]
            else:
                fixed_params[key] = params[key]

        logger.info("%d/%d, parameters: %s" % (num_evaluations,
                                               evaluation_budget,
                                               str(fixed_params)))
        result = target_algorithm(fixed_params)
        evaluations.append(Experiment(params, result))
        num_evaluations += 1

    return min([evaluation[1] for evaluation in evaluations])


def main(args=None):
    args = parse_parameters()
    if args.cwd:
        os.chdir(args.cwd)

    # fh = open(args.params)
    # param_string = fh.read()
    # fh.close()
    # hyperparameters = optimizer_base.parse_hyperparameter_string(
    # param_string)
    # grid = optimizer_base.build_grid(hyperparameters)
    context = setup(args)

    if args.algorithm:
        raise NotImplementedError()
    elif args.cli_target:
        cli_function = optimizer_base.command_line_function
        fn = functools.partial(cli_function, args.cli_target)

    perform_optimization(fn, metalearn_suggest, None, context, sys.maxint,
                         args.number_of_jobs)


if __name__ == "__main__":
    main()
    exit(0)
