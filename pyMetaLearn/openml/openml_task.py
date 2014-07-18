from collections import defaultdict
import cPickle
import os
import sys
import numpy as np

import arff

import pyMetaLearn.openml.manage_openml_data
from pyMetaLearn.openml.openml_dataset import OpenMLDataset

import sklearn.metrics
from sklearn.cross_validation import StratifiedKFold

class OpenMLTask(object):
    def __init__(self, task_id, task_type, data_set_id, target_feature,
                 estimation_procudure_type, data_splits_url,
                 estimation_parameters, evaluation_measure,
                 local_validation_split_file, local_test_split_file):
        # General
        self.task_id = task_id
        self.task_type = task_type
        # Dataset
        self.dataset_id = data_set_id
        self.target_feature = target_feature
        # Estimation
        # TODO: this can become its own class if necessary
        self.estimation_procedure = dict()
        self.estimation_procedure["type"] = estimation_procudure_type
        # TODO: ideally this has the indices for the different splits...but
        # the evaluation procedure 3foldtest/10foldvalid is not available
        self.estimation_procedure["data_splits_url"] = data_splits_url
        self.estimation_procedure["parameters"] = estimation_parameters
        self.estimation_procedure["local_test_split_file"] = local_test_split_file
        self.estimation_procedure["local_validation_split_file"] = local_validation_split_file
        # Evaluation Measure
        self.evaluation_measure = evaluation_measure
        # Predictions
        # TODO: Implement if necessary

    @classmethod
    def from_xml_file(cls, xml_file):
        with open(xml_file, "r") as fh:
            task_xml = fh.read()
        dic = pyMetaLearn.openml.manage_openml_data._xml_to_dict(task_xml)["oml:task"]

        estimation_parameters = dict()
        inputs = dict()
        # Due to the unordered structure we obtain, we first have to extract
        # the possible keys of oml:input; dic["oml:input"] is a list of
        # OrderedDicts
        for input_ in dic["oml:input"]:
            name = input_["@name"]
            inputs[name] = input_

        # Convert some more parameters
        for parameter in inputs["estimation_procedure"]["oml:estimation_procedure"]["oml:parameter"]:
            name = parameter["@name"]
            text = parameter.get("#text", "")
            estimation_parameters[name] = text

        return cls(dic["oml:task_id"], dic["oml:task_type"],
                   inputs["source_data"]["oml:data_set"]["oml:data_set_id"],
                   inputs["source_data"]["oml:data_set"]["oml:target_feature"],
                   inputs["estimation_procedure"]["oml:estimation_procedure"]["oml:type"],
                   inputs["estimation_procedure"]["oml:estimation_procedure"][
                       "oml:data_splits_url"], estimation_parameters,
                   inputs["evaluation_measures"]["oml:evaluation_measures"][
                       "oml:evaluation_measure"], None, None)

    def __str__(self):
        return "OpenMLTask instance.\nTask ID: %s\n" \
               "Task type: %s\nDataset id: %s"\
            % (self.task_id, self.task_type, self.dataset_id)

    def _get_dataset(self):
        # TODO: add the to_lower for the class everywhere its called
        try:
            dataset = pyMetaLearn.openml.manage_openml_data.get_local_dataset(
                self.dataset_id)
        except:
            local_dir = pyMetaLearn.openml.manage_openml_data.get_local_directory()
            dataset_dir = os.path.join(local_dir, "datasets")
            dataset_file = os.path.join(dataset_dir, "did%d.pkl" % self.dataset_id)
            with open(dataset_file) as fh:
                dataset = OpenMLDataset(**cPickle.load(fh))

        return dataset

    def get_dataset_as_pandas(self):
        dataset = self._get_dataset()
        X, Y = dataset.get_pandas(target=self.target_feature.lower())
        return X, Y

    def get_dataset(self):
        # TODO: add possibility to add the scaling etc to the dataset
        # creation routine!
        dataset = self._get_dataset()
        X, Y = dataset.get_npy(target=self.target_feature.lower())
        return X, Y

    def evaluate(self, algo):
        """Evaluate an algorithm on the test data.
        """
        raise NotImplementedError()

    def perform_cv_fold(self, algo, fold, folds):
        """Allows the user to perform cross validation for hyperparameter
        optimization on the training data."""
        # TODO: this is only done for hyperparameter optimization and is not
        # part of the OpenML specification. The OpenML specification would
        # like to have the hyperparameter evaluation inside the evaluate
        # performed by the target algorithm itself. Hyperparameter
        # optimization on the other hand needs these both things to be decoupled
        # For being closer to OpenML one could also call evaluate and pass
        # everything else through kwargs.
        if self.task_type != "Supervised Classification":
            raise NotImplementedError(self.task_type)

        if self.estimation_procedure["type"] != \
                "crossvalidation with holdout":
            raise NotImplementedError(self.estimation_procedure["type"] )

        if self.estimation_procedure["parameters"]["stratified_sampling"] != \
                'true':
            raise NotImplementedError(self.estimation_procedure["parameters"]["stratified_sampling"])

        if self.evaluation_measure != "predictive_accuracy":
            raise NotImplementedError(self.evaluation_measure)

        ########################################################################
        # Test folds
        train_indices, test_indices = self.get_train_test_split()

        ########################################################################
        # Crossvalidation folds
        train_indices, validation_indices = self.get_validation_split(fold)

        X, Y = self.get_dataset()

        algo.fit(X[train_indices], Y[train_indices])

        predictions = algo.predict(X[validation_indices])
        accuracy = sklearn.metrics.accuracy_score(Y[validation_indices], predictions)
        return accuracy

    def get_validation_split(self, fold):
        with open(self.estimation_procedure["local_validation_split_file"]) as fh:
            test_splits = arff.load(fh)

        train_indices = []
        validation_indices = []
        for line in test_splits['data']:
            if line[3] != fold:
                continue
            elif line[0] == 'TRAIN':
                train_indices.append(line[1])
            elif line[0] == 'TEST':
                validation_indices.append(line[1])
            else:
                raise ValueError()

        train_indices = np.array(train_indices)
        validation_indices = np.array(validation_indices)

        return train_indices, validation_indices

    def get_train_test_split(self):
        with open(self.estimation_procedure["local_test_split_file"]) as fh:
            validation_splits = arff.load(fh)

        train_indices = []
        test_indices = []
        for line in validation_splits['data']:
            if line[3] != 0:
                raise NotImplementedError()
            elif line[0] == 'TRAIN':
                train_indices.append(line[1])
            elif line[0] == 'TEST':
                test_indices.append(line[1])
            else:
                raise ValueError()

        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)

        return train_indices, test_indices

    def _get_fold(self, X, Y, fold, folds, shuffle=True):
        fold = int(fold)
        folds = int(folds)
        if fold >= folds:
            raise ValueError((fold, folds))
        if X.shape[0] != Y.shape[0]:
            raise ValueError("The first dimension of the X and Y array must "
                             "be equal.")

        if shuffle == True:
            rs = np.random.RandomState(42)
            indices = np.arange(X.shape[0])
            rs.shuffle(indices)
            Y = Y[indices]

        kf = StratifiedKFold(Y, n_folds=folds, indices=True)
        for idx, split in enumerate(kf):
            if idx == fold:
                break

        if shuffle == True:
            return indices[split[0]], indices[split[1]]
        return split