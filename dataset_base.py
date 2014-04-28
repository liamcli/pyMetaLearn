import cPickle
import os

import pyMetaLearn.openml.manage_openml_data
#import pyMetaLearn.skdata_.skdata_adapter

class DatasetBase(object):
    def __init__(self):
        self.repositories = dict()

        # All repositories are hard-coded so far...
        self.repositories["OPENML"] = pyMetaLearn.openml.manage_openml_data
        #self.repositories["SKDATA"] = pyMetaLearn.skdata_.skdata_adapter

    def get_repositories(self):
        raise NotImplementedError()

    def get_dataset_from_key(self, key):
        splitted = key.split(":")
        repo_name = splitted[0]
        name = splitted[1]
        if len(splitted) == 2 and repo_name in self.repositories:
            repo = self.repositories[repo_name]
            dataset = repo.get_local_dataset(name)
            return dataset

            """"
            if repo == "OPENML":
                dataset = repo.get_local_dataset(name)
                return dataset
            if repo == "SKDATA":
                raise NotImplementedError()
                # TODO: would be cool if both repos had kind of the same
                # interface
                dataset = repo.get_local_dataset(name)
                return dataset
            else:
                raise NotImplementedError("Currently, only %s are supported, "
                    "not %s" % (str(self.repositories.keys()), repo))
            """

        else:
            raise ValueError("A dataset must be described by a key in the "
                             "following form: REPOSITORY:name")

    def get_datasets_from_keys(self, keys):
        raise NotImplementedError()