from collections import namedtuple, defaultdict
import itertools
import os
import Queue
import sys
import time

import numpy as np
import pandas as pd
import scipy
import scipy.misc
import scipy.optimize
import sklearn.ensemble
import sklearn.utils

import HPOlib.benchmark_util as benchmark_util
import pyMetaLearn.metalearning.create_datasets as create_datasets
from pyMetaLearn.metalearning.meta_base import Run, MetaBase
import pyMetaLearn.openml.manage_openml_data


class KNearestDatasets(object):
    def __init__(self, distance='l1', random_state=None, distance_kwargs=None):
        self.distance = distance
        self.model = None
        self.distance_kwargs = distance_kwargs
        self.metafeatures = None
        self.runs = None
        self.best_hyperparameters_per_dataset = None
        self.random_state = sklearn.utils.check_random_state(random_state)

        if self.distance_kwargs is None:
            self.distance_kwargs = {}

    def fit(self, metafeatures, runs):
        assert isinstance(metafeatures, pd.DataFrame)
        assert metafeatures.values.dtype == np.float64
        assert np.isfinite(metafeatures.values).all()
        assert isinstance(runs, dict)
        assert len(runs) == metafeatures.shape[0], (len(runs), metafeatures
                                                    .shape[0])

        self.metafeatures = metafeatures
        self.runs = runs

        # for each dataset, sort the runs according to their result
        best_hyperparameters_per_dataset = {}
        for dataset_name in runs:
            best_hyperparameters_per_dataset[dataset_name] = \
                sorted(runs[dataset_name], key=lambda t: t.result)
        self.best_hyperparameters_per_dataset = best_hyperparameters_per_dataset

        if self.distance == 'learned':
            # TODO: instead of a random forest, the user could provide a generic
            # import call with which it is possible to import a class which
            # implements the sklearn fit and predict function...
            self.distance_kwargs['random_state'] = self.random_state
            sys.stderr.write("Going to use the following RF hyperparameters\n")
            sys.stderr.write(str(self.distance_kwargs) + "\n")
            sys.stderr.flush()
            self.model = LearnedDistanceRF(**self.distance_kwargs)
            return self.model.fit(metafeatures, runs)
        elif self.distance == 'mfs_l1':
            # This implements metafeature selection as described by Matthias
            # Reif in 'Metalearning for evolutionary parameter optimization
            # of classifiers'
            # TODO: should this really in the model variable or not rather
            # something called filter?
            self.model = MetaFeatureSelection(**self.distance_kwargs)
            return self.model.fit(metafeatures, runs)
        elif self.distance == 'mfw_l1':
            self.model = MetaFeatureSelection(mode='weight', **self.distance_kwargs)
            return self.model.fit(metafeatures, runs)
        elif self.distance not in ['l1', 'l2', 'random']:
            raise NotImplementedError(self.distance)

    def kNearestDatasets(self, x, k=1):
        # k=-1 return all datasets sorted by distance
        assert type(x) == pd.Series
        if k < -1 or k == 0:
            raise ValueError('Number of neighbors k cannot be zero or negative.')
        distances = self._calculate_distances_to(x)
        sorted_distances = sorted(distances.items(), key=lambda t: t[1])
        # sys.stderr.write(str(sorted_distances))
        # sys.stderr.write("\n")
        # sys.stderr.flush()

        if k == -1:
            k = len(sorted_distances)
        return sorted_distances[:k]

    def kBestSuggestions(self, x, k=1, exclude_double_configurations=True):
        assert type(x) == pd.Series
        if k < -1 or k == 0:
            raise ValueError('Number of neighbors k cannot be zero or negative.')
        sorted_distances = self.kNearestDatasets(x, -1)
        kbest = []

        if exclude_double_configurations:
            added_hyperparameters = set()
            for dataset, distance in sorted_distances:
                best_hyperparameters = self.best_hyperparameters_per_dataset[dataset][0].params
                if str(best_hyperparameters) not in added_hyperparameters:
                    added_hyperparameters.add(str(best_hyperparameters))
                    kbest.append((dataset, distance, best_hyperparameters))
                if k != -1 and len(kbest) >= k:
                    break
        else:
            for dataset, distance in sorted_distances:
                best_hyperparameters = self.best_hyperparameters_per_dataset[dataset][0].params
                kbest.append((dataset, distance, best_hyperparameters))

        if k == -1:
            k = len(kbest)
        return kbest[:k]

    def _calculate_distances_to(self, other):
        # TODO can this calculate distances to itself?
        distances = {}
        assert isinstance(other, pd.Series)
        assert other.values.dtype == np.float64
        assert np.isfinite(other.values).all()

        if other.name in self.metafeatures.index:
            raise ValueError("You are trying to calculate the distance to a "
                             "dataset which is in your base data.")

        if self.distance in ['l1', 'l2', 'mfs_l1', 'mfw_l1']:
            metafeatures, other = self._scale(self.metafeatures, other)
        else:
            metafeatures = self.metafeatures

        for idx, mf in metafeatures.iterrows():
            dist = self._calculate_distance(mf, other)
            distances[mf.name] = dist

        return distances

    def _scale(self, metafeatures, other):
        assert isinstance(other, pd.Series)
        assert other.values.dtype == np.float64
        scaled_metafeatures = metafeatures.copy(deep=True)
        other = other.copy(deep=True)

        mins = scaled_metafeatures.min()
        maxs = scaled_metafeatures.max()
        # I also need to scale the target dataset meta features...
        mins = pd.DataFrame(data=[mins, other]).min()
        maxs = pd.DataFrame(data=[maxs, other]).max()
        scaled_metafeatures = (scaled_metafeatures - mins) / (maxs - mins)
        other = (other -mins) / (maxs - mins)
        return scaled_metafeatures, other

    def _calculate_distance(self, d1, d2):
        distance_fn = getattr(self, "_" + self.distance)
        return distance_fn(d1, d2)

    def _l1(self, d1, d2):
        """http://en.wikipedia.org/wiki/Norm_%28mathematics%29#Taxicab_norm_or_Manhattan_norm"""
        return np.sum(abs(d1 - d2))

    def _l2(self, d1, d2):
        """http://en.wikipedia.org/wiki/Norm_%28mathematics%29#p-norm"""
        return np.sqrt(np.sum((d1 - d2)**2))

    def _random(self, d1, d2):
        return self.random_state.random_sample()

    def _learned(self, d1, d2):
        model = self.model
        x = np.hstack((d1, d2))

        predictions = model.predict(x)
        # Predictions are between -1 and 1, -1 indicating a negative correlation.
        # Since we evaluate the dataset with the smallest distance, we would
        # evaluate the dataset with the most negative correlation
        #logger.info(predictions)
        #logger.info(predictions[0] * -1)
        return (predictions[0] * -1) + 1

    def _mfs_l1(self, d1, d2):
        d1 = d1.copy() * self.model.weights
        d2 = d2.copy() * self.model.weights
        return self._l1(d1, d2)

    def _mfw_l1(self, d1, d2):
        return self._mfs_l1(d1, d2)


class LearnedDistanceRF(object):
    def __init__(self, n_estimators=100, max_features=0.15,
                 min_samples_split=5, min_samples_leaf=5, n_jobs=1,
                 random_state=None, oob_score=False):
        if isinstance(random_state, str):
            random_state = int(random_state)
        rs = sklearn.utils.check_random_state(random_state)
        rf = sklearn.ensemble.RandomForestRegressor(
            n_estimators=int(n_estimators), max_features=float(max_features),
            min_samples_split=int(min_samples_split), min_samples_leaf=int(min_samples_leaf),
            criterion="mse", random_state=rs, oob_score=oob_score, n_jobs=int(n_jobs))
        self.model = rf

    def fit(self, metafeatures, runs):
        X, Y = self._create_dataset(metafeatures, runs)
        model = self._fit(X, Y)
        return model

    def _create_dataset(self, metafeatures, runs):
        X, Y = create_datasets.create_predict_spearman_rank(
                metafeatures, runs, "permutation")
        return X, Y

    def _fit(self, X, Y):
        self.model.fit(X, Y)
        return self.model

    def predict(self, metafeatures):
        assert isinstance(metafeatures, np.ndarray)
        return self.model.predict(metafeatures)


class MetaFeatureSelection(object):
    def __init__(self, max_number_of_combinations=10, random_state=None,
                 k=1, max_features=0.5, mode='select'):
        self.max_number_of_combinations = max_number_of_combinations
        self.random_state = sklearn.utils.check_random_state(random_state)
        self.k = k
        self.max_features = max_features
        self.weights = None
        self.mode= mode

    def fit(self, metafeatures, runs):
        self.datasets = metafeatures.index
        self.all_other_datasets = {}             # For faster indexing
        self.all_other_runs = {}               # For faster indexing
        self.parameter_distances = defaultdict(dict)
        self.best_hyperparameters_per_dataset = {}
        self.mf_names = metafeatures.columns
        self.kND = KNearestDatasets(distance='l1')

        for dataset in self.datasets:
            self.all_other_datasets[dataset] = \
                pd.Index([name for name in self.datasets if name != dataset])

        for dataset in self.datasets:
            self.all_other_runs[dataset] = \
                {key: runs[key] for key in runs if key != dataset}

        for dataset in self.datasets:
            self.best_hyperparameters_per_dataset[dataset] = \
                sorted(runs[dataset], key=lambda t: t.result)[0]

        for d1, d2 in itertools.combinations(self.datasets, 2):
            hps1 = self.best_hyperparameters_per_dataset[d1]
            hps2 = self.best_hyperparameters_per_dataset[d2]
            keys = set(hps1.params.keys())
            keys.update(hps2.params.keys())
            dist = 0
            for key in keys:
                # TODO: test this; it can happen that string etc occur
                try:
                    p1 = float(hps1.params.get(key, 0))
                    p2 = float(hps2.params.get(key, 0))
                    dist += abs(p1 - p2)
                except:
                    dist += 0 if hps1.params.get(key, 0) == \
                                 hps2.params.get(key, 0) else 1

                #dist += abs(hps1.params.get(key, 0) - hps2.params.get(key, 0))
            self.parameter_distances[d1][d2] = dist
            self.parameter_distances[d2][d1] = dist

        if self.mode == 'select':
            self.weights = self._fit_binary_weights(metafeatures)
        elif self.mode == 'weight':
            self.weights = self._fit_weights(metafeatures)

        sys.stderr.write(str(self.weights))
        sys.stderr.write('\n')
        sys.stderr.flush()
        return self.weights

    def _fit_binary_weights(self, metafeatures):
        best_selection = None
        best_distance = sys.maxint

        for i in range(2, int(np.round(len(self.mf_names) * self.max_features))):
            sys.stderr.write(str(i))
            sys.stderr.write('\n')
            sys.stderr.flush()

            combinations = []
            for j in range(self.max_number_of_combinations):
                combination = []
                target = i
                maximum = len(self.mf_names)
                while len(combination) < target:
                    random = self.random_state.randint(maximum)
                    name = self.mf_names[random]
                    if name not in combination:
                        combination.append(name)

                combinations.append(pd.Index(combination))

            for j, combination in enumerate(combinations):
                dist = 0
                for dataset in self.datasets:
                    hps = self.best_hyperparameters_per_dataset[dataset]
                    self.kND.fit(metafeatures.loc[self.all_other_datasets[
                        dataset], combination], self.all_other_runs[dataset])
                    nearest_datasets = self.kND.kBestSuggestions(
                        metafeatures.loc[dataset, np.array(combination)], self.k)
                    for nd in nearest_datasets:
                        # print "HPS", hps.params, "nd", nd[2]
                        dist += self.parameter_distances[dataset][nd[0]]

                if dist < best_distance:
                    best_distance = dist
                    best_selection = combination

        weights = dict()
        for metafeature in metafeatures:
            if metafeature in best_selection:
                weights[metafeature] = 1
            else:
                weights[metafeature] = 0
        return pd.Series(weights)

    def _fit_weights(self, metafeatures):
        best_weights = None
        best_distance = sys.maxint

        def objective(weights):
            dist = 0
            for dataset in self.datasets:
                self.kND.fit(metafeatures.loc[self.all_other_datasets[
                    dataset], :] * weights, self.all_other_runs[dataset])
                nearest_datasets = self.kND.kBestSuggestions(
                    metafeatures.loc[dataset, :] * weights, self.k)
                for nd in nearest_datasets:
                    dist += self.parameter_distances[dataset][nd[0]]

            return dist

        for i in range(10):
            w0 = np.ones((len(self.mf_names, ))) * 0.5 + \
                (np.random.random(size=len(self.mf_names)) - 0.5) * i / 10
            bounds = [(0, 1) for idx in range(len(self.mf_names))]

            res = scipy.optimize.minimize\
                (objective, w0, bounds=bounds, method='L-BFGS-B',
                 options={'disp': True})

            if res.fun < best_distance:
                best_distance = res.fun
                best_weights = pd.Series(res.x, index=self.mf_names)

        return best_weights



################################################################################
# Stuff for offline hyperparameter optimization of the Distance RF

def _validate_rf_without_one_dataset(X, Y, rf, dataset_name):
    X_train, Y_train, X_valid, Y_valid = split_for_loo(X, Y, dataset_name)
    predictions = rf.model.predict(X_valid)
    rho =  scipy.stats.kendalltau(Y_valid, predictions)[0]
    mae = sklearn.metrics.mean_absolute_error(predictions, Y_valid)
    mse = sklearn.metrics.mean_squared_error(predictions, Y_valid)
    return mae, mse, rho


def train_rf_without_one_dataset(X, Y, rf, dataset_name):
    # Pay attention, this is not for general sklearn models, but for adapted
    # models...
    X_train, Y_train, X_valid, Y_valid = split_for_loo(X, Y, dataset_name)
    rf.model.fit(X_train, Y_train)
    return rf


def split_for_loo(X, Y, dataset_name):
    train = []
    valid = []
    for cross in X.index:
        if dataset_name in cross:
            valid.append(cross)
        else:
            train.append(cross)

    X_train = X.loc[train].values
    Y_train = Y.loc[train].values.reshape((-1,))
    X_valid = X.loc[valid].values
    Y_valid = Y.loc[valid].values.reshape((-1,))
    return X_train, Y_train, X_valid, Y_valid


# TODO: this file has too many tasks, move the optimization of the distance
# function and the forward selection to some different files,
# maybe generalize these things to work for other models as well...
if __name__ == "__main__":
    """For a given problem train the distance function and return its loss
    value. Arguments:
      * task_files_list
      * experiment_files_list
      * metalearning_directory

    You can also enable forward selection by adding '--forward_selection True'
    You can also enable embedded feature selection by adding '--embedded_selection True'
    You can add '--keep_configurations -preprocessing=None,-classifier=LibSVM
    Sample call: python kND.py --task_files_list /home/feurerm/thesis/experiments/experiment/2014_06_01_AutoSklearn_metalearning/tasks.txt
    --experiments_list /home/feurerm/thesis/experiments/experiment/2014_06_01_AutoSklearn_metalearning/experiments_fold0.txt
    --metalearning_directory /home/feurerm/thesis/experiments/experiment --params -random_state 5
    """
    starttime = time.time()
    args, params = benchmark_util.parse_cli()
    os.chdir(args['metalearning_directory'])
    pyMetaLearn.openml.manage_openml_data.set_local_directory(args['metalearning_directory'])

    with open(args["task_files_list"]) as fh:
         task_files_list = fh.readlines()
    with open(args["experiments_list"]) as fh:
         experiments_list = fh.readlines()

    if 'keep_configurations' in args:
        keep_configurations = args['keep_configurations']
        keep_configurations = keep_configurations.split(',')
        keep_configurations = tuple([tuple(kc.split('=')) for kc in keep_configurations])
    else:
        keep_configurations = None

    meta_base = MetaBase(task_files_list, experiments_list, keep_configurations)
    metafeatures = meta_base.get_all_train_metafeatures_as_pandas()
    runs = meta_base.get_all_runs()

    rf = LearnedDistanceRF(**params)
    X, Y = rf._create_dataset(metafeatures, runs)
    import cPickle
    with open("test.pkl", "w") as fh:
        cPickle.dump((X, Y, metafeatures), fh, -1)

    print "Metafeatures", metafeatures.shape
    print "X", X.shape, np.isfinite(X).all().all()
    print "Y", Y.shape, np.isfinite(Y).all()
    print Y


    metafeature_sets = Queue.Queue()
    if 'forward_selection' in args:
        used_metafeatures = []
        metafeature_performance = []
        print "Starting forward selection ",
        i = 0
        for m1, m2 in itertools.combinations(metafeatures.columns, 2):
            metafeature_sets.put(pd.Index([m1, m2]))
            i += 1
        print "with %d metafeature combinations" % i
    elif 'embedded_selection' in args:
        metafeature_performance = []
        metafeature_sets.put(metafeatures.columns)
    else:
        metafeature_sets.put(metafeatures.columns)


    while not metafeature_sets.empty():
        metafeature_set = metafeature_sets.get()
        metafeature_ranks = defaultdict(float)
        loo_mae = []
        loo_rho = []
        loo_mse = []

        print "###############################################################"
        print "New iteration of FS with:"
        print metafeature_set
        print "Dataset Mae MSE Rho"
        # Leave one out CV
        for idx in range(metafeatures.shape[0]):
            leave_out_dataset = metafeatures.index[idx]
            if 'forward_selection' not in args:
                print leave_out_dataset,

            columns = np.hstack(("0_" + metafeature_set,
                                 "1_" + metafeature_set))
            X_ = X.loc[:,columns]
            rf = train_rf_without_one_dataset(X_, Y, rf, leave_out_dataset)
            mae, mse, rho = _validate_rf_without_one_dataset(X_, Y, rf,
                                                        leave_out_dataset)
            if 'forward_selection' not in args:
                print mae, mse, rho

            loo_mae.append(mae)
            loo_rho.append(rho)
            loo_mse.append(mse)
            mf_importances = [(rf.model.feature_importances_[i], X_.columns[i])
                              for i in range(X_.shape[1])]
            mf_importances.sort()
            mf_importances.reverse()
            for rank, item in enumerate(mf_importances):
                score, mf_name = item
                metafeature_ranks[mf_name] += float(rank)

        mae = np.mean(loo_mae)
        mae_std = np.std(loo_mae)
        mse = np.mean(loo_mse)
        mse_std = np.mean(loo_mse)
        rho = np.mean(loo_rho)
        rho_std = np.std(loo_rho)

        mean_ranks = [(metafeature_ranks[mf_name] / metafeatures.shape[0], mf_name)
                      for mf_name in X.columns]
        mean_ranks.sort()

        # TODO: save only validate-best runs!
        print "MAE", mae, mae_std
        print "MSE", mse, mse_std
        print "Mean tau", rho, rho_std
        duration = time.time() - starttime

        if 'forward_selection' in args:
            metafeature_performance.append((mse, metafeature_set))
            # TODO: this can also be sorted in a pareto-optimal way...

            if metafeature_sets.empty():
                if len(used_metafeatures) == 10:
                    break

                print "#######################################################"
                print "#######################################################"
                print "Adding a new feature to the feature set"
                metafeature_performance.sort()
                print metafeature_performance
                used_metafeatures = metafeature_performance[0][1]
                for metafeature in metafeatures.columns:
                    if metafeature in used_metafeatures:
                        continue
                    # I don't know if indexes are copied
                    tmp = [uaie for uaie in used_metafeatures]
                    tmp.append(metafeature)
                    metafeature_sets.put(pd.Index(tmp))
                metafeature_performance = []

        elif 'embedded_selection' in args:
            if len(metafeature_set) <= 2:
                break

            # Remove a metafeature; elements are (average rank, name);
            # only take the name from index two on
            # because the metafeature is preceeded by the index of the
            # dataset which is either 0_ or 1_
            remove = mean_ranks[-1][1][2:]
            print "Going to remove", remove
            keep = pd.Index([mf_name for mf_name in metafeature_set if
                             mf_name != remove])
            print "I will keep", keep
            metafeature_sets.put(keep)

        else:
            for rank in mean_ranks:
                print rank

    if 'forward_selection' in args:
        metafeature_performance.sort()
        print metafeature_performance
        mse = metafeature_performance[0][0]
    print "Result for ParamILS: %s, %f, 1, %f, %d, %s" % \
        ("SAT", abs(duration), mse, -1, str(__file__))
