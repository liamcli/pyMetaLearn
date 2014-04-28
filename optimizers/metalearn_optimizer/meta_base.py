from collections import OrderedDict
import logging
import pandas as pd
import os
import cPickle

import numpy as np

import pyMetaLearn.metafeatures.metafeatures

logging.basicConfig(format='[%(levelname)s] [%(asctime)s:%(name)s] %('
                           'message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("META_BASE")
logger.setLevel(logging.INFO)


class MetaBase(object):
    def __init__(self, datasets, experiments):
        """Container for dataset metadata and experiment results.

        Constructor arguments:
        - datasets: A list of datasets for which experiments were already
            performed.
        - experiments: A list in which every entry corresponds to one entry
            in datasets. It must contain hyperparameter/return value pairs.

        Auxiliary datasets for the config list can be found with the following
        code snippet:

        for i, directory in enumerate(directories):
        dir_list = directories[:i] + directories[(i + 1):]
        with open(os.path.join(directory, directory + "_other_datasets.txt"), "w") as fh:
            for dir in dir_list:
                fh.write(os.path.abspath(dir) + "\n")
        """
        self.datasets = OrderedDict()
        for ds in datasets:
            self.datasets[ds._name] = ds
        self.experiments = OrderedDict()
        for i, exp in enumerate(experiments):
            name = self.datasets.keys()[i]
            self.experiments[name] = exp

    def get_dataset(self, name):
        """Return dataset attribute"""
        return self.datasets[name]

    def get_datasets(self):
        """Return datasets attribute."""
        return self.datasets

    def get_experiment(self, dataset_name):
        """Return a list with all experiments as params/result pairs for a
        dataset name.
        """
        return self.experiments[dataset_name]

    def get_metadata_as_pandas(self, dataset_name, subset_indices=None,
                               metafeature_subset=None):
        features = self.datasets[dataset_name].get_metafeatures(subset_indices)
        df = pd.Series(data=features, name=dataset_name)

        if metafeature_subset is not None:
            subset = pyMetaLearn.metafeatures.metafeatures.subsets[metafeature_subset]
            df = df.loc[subset]
            if len(df) == 0:
                logger.warn("Warning, empty metadata for ds %s" % dataset_name)

        if not np.isfinite(df.values).all():
            logger.warn(df)
            logger.warn(metafeature_subset)
            raise ValueError("Metafeatures contain non-finite values.")
        return df

    def get_all_metadata_as_pandas(self, subset_indices=None,
                                   metafeature_subset=None):
        """Create a pandas DataFrame for the metadata of all datasets."""
        series = []

        datasets = self.get_datasets()
        for key in datasets:
            series.append(self.get_metadata_as_pandas(key, subset_indices,
                                                      metafeature_subset))

        retval = pd.DataFrame(series)
        return retval
