import cPickle
import os

import pyMetaLearn.openml.manage_openml_data as openml

class DatasetBase(object):
    def __init__(self):
        self.repositories = dict()

        # All repositories are hard-coded so far...
        self.repositories["OPENML"] = openml

    def get_repositories(self):
        raise NotImplementedError()

    def get_dataset_from_key(self, key):
        splitted = key.split(":")
        repo = splitted[0]
        name = splitted[1]
        if len(splitted) == 2:
            if repo in self.repositories:
                dir = self.repositories[repo].get_local_directory()
                dataset_file = os.path.join(dir, name, name + '.pkl')
                with open(dataset_file) as fh:
                    dataset = cPickle.load(fh)
                return dataset
            else:
                raise NotImplementedError("Currently, only %s are supported" %
                                          str(self.repositories.keys()))

        else:
            raise ValueError("A dataset must be described by a key in the "
                             "following form: REPOSITORY:name")

    def get_datasets_from_keys(self, keys):
        raise NotImplementedError()