from collections import OrderedDict, namedtuple, defaultdict
import logging
import pandas as pd
import os
import cPickle

import arff
import numpy as np

import pyMetaLearn.metafeatures.metafeatures
from pyMetaLearn.openml.openml_task import OpenMLTask
from pyMetaLearn.openml.openml_dataset import OpenMLDataset

logging.basicConfig(format='[%(levelname)s] [%(asctime)s:%(name)s] %('
                           'message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("META_BASE")
logger.setLevel(logging.INFO)


Run = namedtuple("Run", ["params", "result"])

class MetaBase(object):
    def __init__(self, task_files, experiments, keep_configurations=None):
        """Container for dataset metadata and experiment results.

        Constructor arguments:
        - tasks: A list of task files for which experiments were already
            performed.
        - experiments: A list in which every entry corresponds to one entry
            in datasets. It must contain a list of Runs.
        - keep_parameters: Specifiy (parameter, value) pairs which should be
            kept. Useful to look only at a subproblem, e.g. in the CASH problem
        """

        # TODO: read the tasks from the task_files
        # TODO: read the experiments from the experiments file
        # TODO: check if the files actually have the same length

        self.local_directory = pyMetaLearn.openml.manage_openml_data\
            .get_local_directory()
        self.dataset_dir = os.path.join(self.local_directory, "datasets")
        self.split_dir = os.path.join(self.local_directory, "splits")
        self.metafeature_dir = os.path.join(self.local_directory, "metafeatures")

        self.tasks = list()
        self.datasets = list()
        self.metafeatures = list()
        self.train_metafeatures = list()
        self.cv_metafeatures = list()
        self.test_splits = list()
        self.cv_splits = list()
        self.runs = list()
        self.cv_runs = list()

        for t in task_files:
            with open(t.strip()) as fh:
                self.tasks.append(OpenMLTask(**cPickle.load(fh)))

            # Get the accompanying dataset
            try:
                dataset = pyMetaLearn.openml.manage_openml_data.get_local_dataset(
                            self.tasks[-1].dataset_id)
            except:
                dataset_file = os.path.join(self.dataset_dir, "did%d.pkl" %
                    self.tasks[-1].dataset_id)
                with open(dataset_file) as fh:
                    dataset = OpenMLDataset(**cPickle.load(fh))

            self.datasets.append(dataset)

            # Get the splits
            test_split_file = self.tasks[-1].estimation_procedure["local_test_split_file"]
            train_split = []
            test_split = []
            with open(test_split_file) as fh:
                split_file = arff.load(fh)
                for line in split_file['data']:
                    if line[2] != 0:
                        raise NotImplementedError("No repeats implemented so far")
                    if line[3] != 0:
                        raise NotImplementedError("No folds for testing so far")
                    if line[0] == "TRAIN":
                        train_split.append(line[1])
                    elif line[0] == "TEST":
                        test_split.append(line[1])
                    else:
                        raise ValueError("Illegal value %s" % line[0])
            self.test_splits.append((train_split, test_split))

            """
            validation_split_file = self.tasks[-1].estimation_procedure["local_validation_split_file"]
            splits_per_fold = defaultdict((list, list))
            with open(validation_split_file) as fh:
                split_file = arff.load(fh)
                for line in split_file['data']:
                    if line[2] != 0:
                        raise NotImplementedError("No repeats implemented so far")
                    if line[0] == "TRAIN":
                        splits_per_fold[line[3]][0].append(line[1])
                    elif line[0] == "TEST":
                        splits_per_fold[line[3]][0].append(line[1])
                    else:
                        raise ValueError("Illegal value %s" % line[0])
            self.cv_splits.append(splits_per_fold)
            """
            # Get the metafeatures
            self.metafeatures.append(dataset.get_metafeatures())
            self.train_metafeatures.append(dataset.get_metafeatures(test_split_file))
            # self.cv_metafeatures.append(dataset.get_metafeatures(
            # validation_split_file))

        for i, exp in enumerate(experiments):
            if not os.path.exists(exp.strip()):
                print "Cannot find %s" % exp
                runs = []
                cv_runs = []
            else:
                with open(exp.strip()) as fh:
                    runs = self.read_experiment_pickle(fh, keep_configurations)
                    fh.seek(0)
                    cv_runs = self.read_folds_from_experiment_pickle(fh, keep_configurations)

                    if keep_configurations is not None:
                        print "Kept %d configurations for %s" % (len(runs), exp.strip())

            self.runs.append(runs)
            self.cv_runs.append(cv_runs)

        class dummy(object):
            pass

        self.dicts = dummy()
        self.dicts.tasks = OrderedDict()
        self.dicts.datasets = OrderedDict()
        self.dicts.test_splits = OrderedDict()
        # self.dicts.validation_splits = OrderedDict()
        self.dicts.metafeatures = OrderedDict()
        self.dicts.train_metafeatures = OrderedDict()
        self.dicts.cv_metafeatures = OrderedDict()
        self.dicts.runs = OrderedDict()
        self.dicts.cv_runs = OrderedDict()
        for i, dataset in enumerate(self.datasets):
            self.dicts.tasks[dataset._name] = self.tasks[i]
            self.dicts.datasets[dataset._name] = self.datasets[i]
            self.dicts.test_splits[dataset._name] = self.test_splits[i]
            # self.dicts.validation_splits[dataset._name] = self.cv_splits[i]
            self.dicts.metafeatures[dataset._name] = self.metafeatures[i]
            self.dicts.train_metafeatures[dataset._name] = self.train_metafeatures[i]
            # self.dicts.cv_metafeatures[dataset._name] = self
            # .cv_metafeatures[i]
            self.dicts.runs[dataset._name] = self.runs[i]
            self.dicts.cv_runs[dataset._name] = self.cv_runs[i]

    def get_dataset(self, name):
        """Return dataset attribute"""
        return self.dicts.datasets[name]

    def get_datasets(self):
        """Return datasets attribute."""
        return self.dicts.datasets

    def get_runs(self, dataset_name):
        """Return a list of all runs for a dataset."""
        return self.dicts.runs[dataset_name]

    def get_all_runs(self):
        """Return a dictionary with a list of all runs"""
        return self.dicts.runs

    def get_cv_runs(self, dataset_name):
        """Return a list of all runs. Each element is a list of runs for the
        crossvalidation fold"""
        return self.dicts.cv_runs[dataset_name]

    def get_metafeatures_as_pandas(self, dataset_name, split_file_name=None,
                               metafeature_subset=None):
        mf = self.dicts.datasets[dataset_name].get_metafeatures(split_file_name)
        df = pd.Series(data=mf, name=dataset_name)

        if metafeature_subset is not None:
            subset = pyMetaLearn.metafeatures.metafeatures.subsets[metafeature_subset]
            df = df.loc[subset]
            if len(df) == 0:
                logger.warn("Warning, empty metadata for ds %s" % dataset_name)

        assert df.values.dtype == np.float64
        if not np.isfinite(df.values).all():
            logger.warn(df)
            logger.warn(metafeature_subset)
            raise ValueError("Metafeatures for dataset %s contain non-finite "
                             "values." % dataset_name)
        return df

    def get_metafeatures_times_as_pandas(self, dataset_name,
            split_file_name=None, metafeature_subset=None):
        mf, times = self.dicts.datasets[dataset_name].get_metafeatures(
            split_file_name, return_times=True, return_helper_functions=True)
        df = pd.Series(data=times, name=dataset_name)

        if metafeature_subset is not None:
            subset = pyMetaLearn.metafeatures.metafeatures.subsets[metafeature_subset]
            df = df.loc[subset]
            if len(df) == 0:
                logger.warn("Warning, empty metadata for ds %s" % dataset_name)

        def check_finiteness(_df):
            assert df.values.dtype == np.float64
            if not np.isfinite(df.values).all():
                logger.warn(df)
                logger.warn(metafeature_subset)
                raise ValueError("Metafeatures for dataset %s contain non-finite "
                                 "values." % dataset_name)

        if type(df) is dict:
            for key in df:
                check_finiteness(df[key])
        if df.values.dtype == np.float64:
            check_finiteness(df)
        else:
            raise NotImplementedError(type(df))

        return df

    def get_all_metafeatures_as_pandas(self, metafeature_subset=None):
        """Create a pandas DataFrame for the metadata of all datasets."""
        series = []

        for key in self.dicts.datasets:
            series.append(self.get_metafeatures_as_pandas(key,
                          metafeature_subset=metafeature_subset))

        retval = pd.DataFrame(series)
        return retval

    def get_all_metafeatures_times_as_pandas(self, metafeature_subset=None):
        """Create a pandas DataFrame for the metadata of all datasets."""
        series = []

        for key in self.dicts.datasets:
            series.append(self.get_metafeatures_times_as_pandas(key,
                          metafeature_subset=metafeature_subset))

        retval = pd.DataFrame(series)
        return retval

    def get_train_metafeatures_as_pandas(self, dataset_name, subset=None):
        split_file_name = self.dicts.tasks[dataset_name].\
            estimation_procedure["local_test_split_file"]
        return self.get_metafeatures_as_pandas(dataset_name, split_file_name,
                                          subset)

    def get_train_metafeatures_times_as_pandas(self, dataset_name, subset=None):
        split_file_name = self.dicts.tasks[dataset_name].\
            estimation_procedure["local_test_split_file"]
        return self.get_metafeatures_times_as_pandas(dataset_name,
            split_file_name, subset)

    def get_all_train_metafeatures_as_pandas(self, subset=None):
        series = []

        for key in self.dicts.datasets:
            split_file = self.dicts.tasks[key]. \
                estimation_procedure["local_test_split_file"]
            series.append(self.get_metafeatures_as_pandas(key, split_file, subset))

        retval = pd.DataFrame(series)
        return retval

    def get_all_train_metafeatures_times_as_pandas(self, subset=None):
        series = []

        for key in self.dicts.datasets:
            split_file = self.dicts.tasks[key]. \
                estimation_procedure["local_test_split_file"]
            series.append(self.get_metafeatures_times_as_pandas(
                key, split_file, subset))

        retval = pd.DataFrame(series)
        return retval

    """
    def get_cv_metafeatures_as_pandas(...

    def get_all_cv_metafeatures_as_pandas(...
    """

    def read_experiment_pickle(self, fh, keep_configurations=None):
        runs = list()
        trials = cPickle.load(fh)
        for trial in trials["trials"]:
            add = True

            if keep_configurations is not None:
                # Keep parameters must be a list of tuples!
                params = trial["params"]
                for key, value in keep_configurations:
                    # Work around annoying HPOlib bug!
                    if str(params['-' + key]) != str(value):
                        add = False
                        break

            if not add:
                continue
            runs.append(Run(trial["params"], trial["result"]))

        return runs

    def read_folds_from_experiment_pickle(self, fh, keep_configurations):
        runs = list()
        trials = cPickle.load(fh)
        for trial in trials["trials"]:
            add = True

            if keep_configurations is not None:
                params = trial["params"]
                # Keep parameters must be a list of tuples!
                for key, value in keep_configurations:
                    # Work around annoying HPOlib bug!
                    if str(params['-' + key]) != str(value):
                        add = False
                        break

            if not add:
                continue
            runs.append(list())
            ir = trial["instance_results"]
            for fold in range(len(ir)):
                runs[-1].append(Run(trial["params"], ir[fold]))
        return runs


