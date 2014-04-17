import sys
import numpy as np

import pyMetaLearn.openml.manage_openml_data

import sklearn.metrics
from sklearn.cross_validation import StratifiedKFold

class OpenMLTask(object):
    def __init__(self, task_id, task_type, data_set_id, target_feature,
                 estimation_procudure_type, data_splits_url,
                 estimation_parameters, evaluation_measure):
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
        for parameter in dic["oml:input"][0]["oml:estimation_procedure"]["oml:parameter"]:
            name = parameter["@name"]
            text = parameter.get("#text", "")
            estimation_parameters[name] = text

        return cls(dic["oml:task_id"], dic["oml:task_type"],
                   dic["oml:input"][2]["oml:data_set"]["oml:data_set_id"],
                   dic["oml:input"][2]["oml:data_set"]["oml:target_feature"],
                   dic["oml:input"][0]["oml:estimation_procedure"]["oml:type"],
                   dic["oml:input"][0]["oml:estimation_procedure"][
                       "oml:data_splits_url"], estimation_parameters,
                   dic["oml:input"][1]["oml:evaluation_measures"][
                       "oml:evaluation_measure"])

    def __str__(self):
        return "OpenMLTask instance.\nTask ID: %s\n" \
               "Task type: %s\nDataset id: %s"\
            % (self.task_id, self.task_type, self.dataset_id)

    def get_dataset(self):
        dataset = pyMetaLearn.openml.manage_openml_data.get_local_dataset(
            self.dataset_id)
        X, Y = dataset.get_npy(target=self.target_feature)
        return X, Y

    def get_train_and_test_set(self, X=None, Y=None):
        if X is None and Y is None:
            # TODO: at some point get the split from OpenML
            X, Y = self.get_dataset()
            rs = np.random.RandomState(42)
            indices = np.arange(X.shape[0])
            rs.shuffle(indices)
            X = X[indices]
            Y = Y[indices]
        elif X is None or Y is None:
            raise NotImplementedError()

        test_folds = self.estimation_procedure["parameters"]["test_folds"]
        test_fold = self.estimation_procedure["parameters"]["test_fold"]
        print "Tests folds", test_fold, test_folds

        split = self._get_fold(X, Y, fold=test_fold, folds=test_folds)
        X_train = X[split[0]]
        X_test = X[split[1]]
        Y_train = Y[split[0]]
        Y_test = Y[split[1]]

        return (X_train, X_test, Y_train, Y_test)

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
            raise NotImplementedError()

        if self.estimation_procedure["type"] != \
                "crossvalidation with crossvalidation holdout":
            raise NotImplementedError()

        if self.estimation_procedure["parameters"]["stratified_sampling"] != \
                'true':
            raise NotImplementedError()

        if self.evaluation_measure != "predictive_accuracy":
            raise NotImplementedError()

        ########################################################################
        # Test folds
        X_train, X_test, Y_train, Y_test = self.get_train_and_test_set()

        ########################################################################
        # Crossvalidation folds
        train_mask, valid_mask = self._get_fold(X_train, Y_train, fold, folds)
        data = dict()
        data["train_X"] = X_train[train_mask]
        data["train_Y"] = Y_train[train_mask]
        data["valid_X"] = X_train[valid_mask]
        data["valid_Y"] = Y_train[valid_mask]
        data["test_X"] = X_test
        data["test_Y"] = Y_test

        algo.fit(data["train_X"], data["train_Y"])

        predictions = algo.predict(data["valid_X"])
        accuracy = sklearn.metrics.accuracy_score(data["valid_Y"], predictions)
        return accuracy


    def _get_fold(self, X, Y, fold, folds):
        fold = int(fold)
        folds = int(folds)
        if fold >= folds:
            raise ValueError((fold, folds))
        # do stratified cross validation, like OpenML does according to the MySQL
        # dump.
        # print fold, "/", folds
        kf = StratifiedKFold(Y, n_folds=folds, indices=True)
        for idx, split in enumerate(kf):
            if idx == fold:
                return split
