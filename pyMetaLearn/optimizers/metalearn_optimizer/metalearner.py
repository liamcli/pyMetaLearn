import argparse
import ast
import cPickle
from collections import namedtuple, OrderedDict
import logging
import functools
import numpy as np
import os
import sys

import sklearn.utils

from pyMetaLearn.metalearning.meta_base import MetaBase, Run
from pyMetaLearn.metalearning.kND import KNearestDatasets
from pyMetaLearn.openml.openml_task import OpenMLTask
import pyMetaLearn.optimizers.optimizer_base as optimizer_base
import pyMetaLearn.openml.manage_openml_data

logging.basicConfig(format='[%(levelname)s] [%(asctime)s:%(name)s] %('
                           'message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("METALEARNING")
logger.setLevel(logging.INFO)

def test_function(params):
    return np.random.random()


class MetaLearningOptimizer(object):
    def __init__(self, task_file, task_files_list, experiments_file_list,
                 openml_dir, distance='l1', seed=None,
                 use_features='', distance_kwargs=None):
        # Document that this has a state, namely the task_file on which it
        # operates;
        # Does not yet work on cv folds, but they are too expensive to train
        # anyway
        self.task_file = task_file
        self.task_files_list = task_files_list
        self.experiments_file_list = experiments_file_list
        self.openml_dir = openml_dir
        self.distance = distance
        self.seed = seed
        self.use_features = use_features
        self.distance_kwargs = distance_kwargs

        self.meta_base = MetaBase(task_files_list, experiments_file_list)

    def perform_sequential_optimization(self, target_algorithm=test_function,
                                        time_budget=sys.maxint,
                                        evaluation_budget=sys.maxint):
        time_taken = 0
        num_evaluations = 0
        history = []

        logger.info("Taking distance measure %s" % self.distance)
        while True:
            if time_taken >= time_budget:
                logger.info("Reached time budget. Exiting optimization.")
                break
            if num_evaluations >= evaluation_budget:
                logger.info("Reached maximum number of evaluations. Exiting "
                            "optimization.")
                break

            params = self.metalearning_suggest(history)

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
            history.append(Run(params, result))
            num_evaluations += 1

        return min([run.result for run in history])

    def metalearning_suggest_all(self, exclude_double_configurations=True):
        """Return a list of the best hyperparameters of neighboring datasets"""
        neighbors = self._learn(exclude_double_configurations)
        hp_list = []
        for neighbor in neighbors:
            logger.info("%s %s %s" % (neighbor[0], neighbor[1], neighbor[2]))
            hp_list.append(neighbor[2])
        return hp_list

    def metalearning_suggest(self, history):
        """Suggest the next most promising hyperparameters which were not yet evaluated"""
        neighbors = self._learn()
        # Iterate over all datasets which are sorted ascending by distance
        for idx, neighbor in enumerate(neighbors):
            already_evaluated = False
            # Check if that dataset was already evaluated
            for run in history:
                # If so, return to the outer loop
                if neighbor[2] == run.params:
                    already_evaluated = True
                    break
            if not already_evaluated:
                logger.info("Nearest dataset with hyperparameters of best value "
                            "not evaluated yet is %s with a distance of %f" %
                            (neighbor[0], neighbor[1]))
                return neighbor[2]
        raise StopIteration("No more values available.")

    def _learn(self, exclude_double_configurations=True):
        dataset_metafeatures, all_other_metafeatures = self._get_metafeatures()

        # In case that we learn our distance function, get the parameters for
        #  the random forest
        if self.distance_kwargs:
            rf_params = ast.literal_eval(self.distance_kwargs)
        else:
            rf_params = None

        # To keep the distance the same in every iteration, we create a new
        # random state
        random_state = sklearn.utils.check_random_state(self.seed)
        kND = KNearestDatasets(distance=self.distance,
                               random_state=random_state,
                               distance_kwargs=rf_params)

        runs = dict()
        for name in all_other_metafeatures.index:
            runs[name] = self.meta_base.get_runs(name)
        kND.fit(all_other_metafeatures, runs)
        return kND.kBestSuggestions(dataset_metafeatures, k=-1,
            exclude_double_configurations=exclude_double_configurations)

    def _get_metafeatures(self):
        """This is inside an extra function for testing purpose"""
        # Load the task
        with open(self.task_file) as fh:
            task_args = cPickle.load(fh)
        task = OpenMLTask(**task_args)
        dataset = pyMetaLearn.openml.manage_openml_data.get_local_dataset(task.dataset_id)

        all_metafeatures = self.meta_base.get_all_train_metafeatures_as_pandas()
        if self.use_features is not None and \
                (type(self.use_features) != str or self.use_features != ''):
            logger.warn("Going to keep the following features %s",
                    str(self.use_features))
            if type(self.use_features) == str:
                use_features = self.use_features.split(",")
            elif type(self.use_features) in (list, np.ndarray):
                use_features = self.use_features
            else:
                raise NotImplementedError(type(self.use_features))
            keep = [col for col in all_metafeatures.columns if col in use_features]
            all_metafeatures = all_metafeatures.loc[:,keep]

        return self._split_metafeature_array(dataset._name, all_metafeatures)

    def _split_metafeature_array(self, dataset_name, metafeatures):
        """Split the metafeature array into dataset metafeatures and all other.

        This is inside an extra function for testing purpose.
        """
        dataset_metafeatures = metafeatures.ix[dataset_name].copy()
        metafeatures = metafeatures[metafeatures.index != dataset_name]
        return dataset_metafeatures, metafeatures

    def read_task_list(self, fh):
        dataset_filenames = list()
        for line in fh:
            line = line.replace("\n", "")
            if line:
                dataset_filenames.append(line)
            else:
                raise ValueError("Blank lines in the task list are not "
                                 "supported.")
        return dataset_filenames

    def read_experiments_list(self, fh):
        experiments_list = list()
        for line in fh.readlines():
            experiments_list.append(line.split())
        return experiments_list


def parse_parameters(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("task_file", type=str,
                        help="The task which should be optimized.")
    parser.add_argument("task_files_list", type=str,
                        help="A list with all task files for which "
                             "should be considered for metalearning")
    parser.add_argument("experiment_files_list", type=str,
                        help="A list with all experiment pickles which "
                             "should be considered for metalearning")
    parser.add_argument("metalearning_directory", type=str,
                        help="A directory with the metalearning datastructure")
    parser.add_argument("-d", "--distance_measure", type=str, default='l1',
                        choices=['l1', 'l2', 'learned', 'random'])
    parser.add_argument("--distance_keep_features", type=str, default='',)
    parser.add_argument("--cli_target")
    # parser.add_argument("-p", "--params", required=True)
    parser.add_argument("--cwd", type=str)
    parser.add_argument("--number_of_jobs", required=True, type=int,
                        default=50)
    parser.add_argument("-s", "--seed", type=int, default=1)
    args = parser.parse_args(args=args)
    return args


def main():
    args = parse_parameters()
    if args.cwd:
        os.chdir(args.cwd)

    # TODO check if the directory contains a valid directory structure!
    # No, don't as we're inside an experiment directory; check if the openml
    # directory contains everything which is necessary!
    pyMetaLearn.openml.manage_openml_data.set_local_directory(args.metalearning_directory)

    cli_function = optimizer_base.command_line_function
    fn = functools.partial(cli_function, args.cli_target)

    with open(args.task_files_list) as fh:
         task_filenames = fh.readlines()
    with open(args.experiment_files_list) as fh:
        experiment_filenames = fh.readlines()

    optimizer = MetaLearningOptimizer(args.task_file, task_filenames,
        experiment_filenames, args.cwd, distance=args.distance_measure,
        seed=args.seed, use_features=args.distance_keep_features,
        distance_kwargs=None)
    try:
        optimizer.perform_sequential_optimization(
            target_algorithm=fn,
            evaluation_budget=args.number_of_jobs)
    except StopIteration:
        logger.warning("No more hyperparameter configurations to be chosen "
                       "via metalearning!")


if __name__ == "__main__":
    main()
    exit(0)