from collections import OrderedDict
import cPickle
import glob
import logging
import os
import numpy as np

import sklearn
import sklearn.preprocessing
import sklearn.tree
import sklearn.ensemble
import sklearn.metrics
import sklearn.cross_validation
import sklearn.utils

import rundata
import rundata.smac_extracter

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def find_all_state_runs(state_runs_dir):
    print state_runs_dir
    print os.path.join(state_runs_dir, "state-run1/param-file.txt")
    state_runs = glob.glob(
        os.path.join(state_runs_dir, "state-run") +
        "*")
    state_runs.sort()
    return state_runs


def read_state_run(state_run):
    num_data_points = 0
    state_run_values = OrderedDict()

    paths = rundata.smac_extracter.get_state_run_file_paths(state_run)

    with open(paths["runs_and_results_path"]) as fh:
        runs_and_results = rundata.smac_extracter.parse_runs_and_results(
            fh.read())
        state_run_values["runs_and_results"] = runs_and_results
        logger.info("Found %d runs in %s" % (
            len(runs_and_results), paths["runs_and_results_path"]))
        state_run_values["num_data_points"] = len(runs_and_results)

    with open(paths["paramstrings_path"]) as fh:
        paramstrings = rundata.smac_extracter.parse_paramstrings(fh.read())
        state_run_values["paramstrings"] = paramstrings

    return state_run_values


def create_training_data(parameter_descriptions, state_runs_values):
    num_data_points = 0
    for state_run_values in state_runs_values:
        num_data_points += len(state_run_values["runs_and_results"])

    # Keys are extracted once from the first sample and then compared to all
    # following samples.
    keys = []
    types = []
    label_encoders = dict()
    for key in state_runs_values[0]["paramstrings"][0]:
        keys.append(key)

        if parameter_descriptions[key]["type"] == "categorical":
            label_encoders[key] = sklearn.preprocessing.LabelEncoder(). \
                fit(parameter_descriptions[key]["values"])
            #label_encoder[key] = sklearn.preprocessing.OneHotEncoder().\
            #    fit(parameter_descriptions[key]["values"])

    print "Trying to create array of size (%d, %d)" % (num_data_points, len(keys))
    X = np.zeros((num_data_points, len(keys))) + np.inf
    Y_time = np.zeros(num_data_points) + np.inf
    Y_performance = np.zeros(num_data_points) + np.inf

    sample_idx = 0
    for state_run_values in state_runs_values:
        # Very basic, the same parameter values are in there with different
        # performance targets
        for idx, run in enumerate(state_run_values["runs_and_results"]):
            Y_performance[sample_idx] = run["Response Value (y)"] / 100
            Y_time[sample_idx] = run["Runtime"] / 100
            params = state_run_values["paramstrings"][int(run["Run History Configuration ID"]) - 1]
            # Check that the keys are in the same order as they are for the very
            # first example seen
            for feature_idx, param_name in enumerate(params):
                if param_name != keys[feature_idx]:
                    raise ValueError("Keys are sorted in a wrong way...")
                elif parameter_descriptions[param_name]["type"] == "categorical":
                    X[sample_idx][feature_idx] =  int(label_encoders[param_name]
                        .transform((params[param_name],))[0])
                else:
                    X[sample_idx][feature_idx] = float(params[param_name])
            sample_idx += 1
            #print "%d/%d" % (sample_idx, num_data_points)

    return X, Y_performance


def save_training_data(X, Y_performance, filename):
    with open(filename, "w") as fh:
        cPickle.dump((X, Y_performance), fh)


def load_training_data(filename):
    with open(filename, "r") as fh:
        X, Y_performance = cPickle.load(fh)
    return X, Y_performance


def calculate_mean_absolute_error(target, prediction):
    return np.mean(abs(target - prediction))


def calculate_mean_squared_error(target, prediction):
    squared_error = (target - prediction) ** 2
    return np.mean(squared_error)


def calculate_root_mean_squared_error(target, prediction):
    return np.sqrt(calculate_mean_squared_error(target, prediction))


def run_smac_workflow(name, directory):

    with open(os.path.join(directory, "state-run1/param-file.txt")) as fh:
        parameter_descriptions = rundata.smac_extracter \
            .read_parameter_names_and_types(fh.read())

    state_runs = find_all_state_runs(directory)

    state_run_values = []
    params = []
    num_data_points = 0
    for state_run in state_runs:
        try:
            state_run_values.append(read_state_run(state_run))
            num_data_points += state_run_values[-1]["num_data_points"]
        except ValueError as e:
            logger.warning(e)
            logger.warning("Going to ignore directory %s" % state_run)

    print "We have %d data points" % num_data_points

    X, Y_performance= create_training_data(parameter_descriptions,
                                                  state_run_values)

    save_training_data(X, Y_performance, name + ".pkl")

    keys = []
    for key in state_run_values[0]["paramstrings"][0]:
        keys.append(key)

    X, Y_performance = load_training_data(name + ".pkl")

    print "Training sklearn.tree.DecisionTreeRegressor"

    #for i, key in enumerate(keys):
    #    print "%d/%d: %s, type %s" % (i, 786, key, parameter_descriptions[
    # key]['type'])

    random_state = sklearn.utils.check_random_state(42)
    kf = sklearn.cross_validation.KFold(num_data_points, 4, shuffle=True,
                                        random_state=random_state)

    total_predictions = np.zeros((num_data_points)) + np.inf
    for train_index, test_index in kf:
        train_targets = Y_performance[train_index]
        test_targets = Y_performance[test_index]

        regressor = sklearn.ensemble.RandomForestRegressor(n_estimators=10)
        regressor = regressor.fit(X[train_index], train_targets)

        train_predictions = regressor.predict(X[train_index])
        test_predictions = regressor.predict(X[test_index])
        total_predictions[test_index] = test_predictions
        """
        print "ME train", calculate_mean_absolute_error(train_predictions,
                                                        train_targets)
        print "MSE train", calculate_mean_squared_error(train_predictions,
                                                        train_targets)
        print "RMSE train", calculate_root_mean_squared_error(train_predictions,
                                                        train_targets)
        print "RMSE train", calculate_root_mean_squared_error(
        train_predictions * 100,
                                                        train_targets * 100)
        print "NMSE train", calculate_mean_squared_error(train_predictions,
                                                        train_targets) \
                           / (np.mean(train_predictions) * np.mean(train_targets))

        print


        print "ME test", calculate_mean_absolute_error(test_predictions,
                                                       test_targets)
        print "MSE test", calculate_mean_squared_error(test_predictions,
                                                       test_targets)
        print "RMSE test", calculate_root_mean_squared_error(test_predictions,
                                                       test_targets)
        print "RMSE test", calculate_root_mean_squared_error(test_predictions * 100,
                                                       test_targets * 100)
        print "NMSE test", calculate_mean_squared_error(test_predictions,
                                                        test_targets) \
                           / (np.mean(test_predictions) * np.mean(Y_performance[
                                                                  4000:]))

        print
        print
        """

    print "ME CV", calculate_mean_absolute_error(total_predictions,
                                                        Y_performance)
    print "MSE CV", calculate_mean_squared_error(total_predictions,
                                                    Y_performance)
    print "RMSE CV", calculate_root_mean_squared_error(total_predictions,
                                                    Y_performance)
    print "RMSE CV", calculate_root_mean_squared_error(total_predictions * 100,
                                                    Y_performance * 100)
    print "NMSE CV", calculate_mean_squared_error(total_predictions,
                                                    Y_performance) \
                       / (np.mean(total_predictions) * np.mean(Y_performance))


if __name__ == "__main__":
    base_dir = "/home/feurerm/thesis/datasets/smac_data"
    dataset_directories = OrderedDict([#["smac:abalone",
                                #"SMAC-CV10-Termination-Abalone-0"],
                               ["smac:amazon",
                               "SMAC-CV10-Termination-Amazon-0"],
                   #["smac:car", "SMAC-CV10-Termination-car-0"],
                   ["smac:cifar-10", "SMAC-CV10-Termination-CIFAR-10-0"],
                   ["smac:cifar-10-large",
                    "SMAC-CV10-Termination-CIFAR-10-Large-0"],
                   ["smac:convex", "SMAC-CV10-Termination-Convex"],
                   #["smac:credit", "SMAC-CV10-Termination-Credit-0"],
                   ["smac:dexter", "SMAC-CV10-Termination-Dexter-0"],
                   ["smac:dorothea", "SMAC-CV10-Termination-Dorothea-0"],
                   ["smac:gisette", "SMAC-CV10-Termination-Gisette-0"],
                   ["smac:kdd09", "SMAC-CV10-Termination-KDD09-0"],
                   #["smac:KRvsKP", "SMAC-CV10-Termination-KRvsKP-0"],
                   #["smac:madelon", "SMAC-CV10-Termination-Madelon-0"],
                   ["smac:mnist", "SMAC-CV10-Termination-MNIST-0"],
                   ["smac:mrbi", "SMAC-CV10-Termination-MNISTBackRotImage"],
                   ["smac:secom", "SMAC-CV10-Termination-SECOM-0"],
                   ["smac:semeion", "SMAC-CV10-Termination-SEMEION-0"],
                   ["smac:shuttle", "SMAC-CV10-Termination-Shuttle"]])
                   #["smac:waveform", "SMAC-CV10-Termination-Waveform-0"]])
                   #["smac:winequalitywhite",
                   # "SMAC-CV10-Termination-WineQualityWhite-0"]])
                   #["smac:yeast", "SMAC-CV10-Termination-Yeast-0"]])
    for dataset_directory in dataset_directories:
        run_smac_workflow(dataset_directory, os.path.join(base_dir,
                                                         dataset_directories[
            dataset_directory]))

