from collections import OrderedDict
import pandas as pd
import os
import cPickle

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

    def get_metadata_as_pandas(self, dataset_name):
        features = self.datasets[dataset_name].get_metafeatures()
        df = pd.Series(data=features, name=dataset_name)
        return df

    def get_all_metadata_as_pandas(self):
        """Create a pandas DataFrame for the metadata of all datasets."""
        series = []

        datasets = self.get_datasets()
        for key in datasets:
            series.append(self.get_metadata_as_pandas(key))

        retval = pd.DataFrame(series)
        return retval
