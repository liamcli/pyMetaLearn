import glob
import os

import pyMetaLearn.openml.manage_openml_data

ground_truth_dir = "/home/feurerm/thesis/experiments/2014_04_17" \
    "_gather_new_metadata_from_openml/"

experiments_directory = os.getcwd()
local_directory = pyMetaLearn.openml.manage_openml_data.get_local_directory()
dataset_dir = os.path.join(local_directory, "datasets")

dataset_list = glob.glob(os.path.join(dataset_dir, "did*.xml"))
print dataset_list

for dataset in dataset_list:
    for fold in range(3):
        experiment_dir = os.path.join(experiments_directory, dataset +
                                                             "_fold%s" % fold)
        print experiment_dir
        os.mkdir(experiment_dir)

        ########################################################################
        # Create all directories for the search spaces
        directories = dict()
        optimizers = ["gridsearch", "smac_2_06_01-dev",
                   "hyperopt_august2013_mod", "spearmint_april2013_mod",
                   "random_hyperopt_august2013_mod"]
        for optimizer in optimizers:
            directories[optimizer] = os.path.join(experiment_dir, optimizer)
            os.mkdir(directories[optimizer])

        # Gridsearch
        with open(os.path.join(directories["gridsearch"], "params.pcs"),
                  "w") as fh:
            fh.write("C {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "
                     "10, 11, 13, 14, 15}\n"
                     "gamma {-15, -14, -13, -12, -11, -10 -9, -8, -7, -6, -5, "
                     "-4, -3, -2, -1, 0, 1, 2, 3}")

        # SMAC
        with open(os.path.join(directories["smac_2_06_01-dev"], "params.pcs"),
                  "w") as fh:
            fh.write("C [-5, 15] [1]i\ngamma [-15, 3] [1]i")

        # Hyperopt
        with open(os.path.join(directories["hyperopt_august2013_mod"],
                  "space.py"), "w") as fh:
            fh.write("from hyperopt import hp\n"
                     "from hyperopt.pyll import scope\n\n"
                     "space = {'C': scope.int(hp.quniform('C', -5, 15, 1)),\n'"
                     "gamma': scope.int(hp.quniform('gamma', -15, 3, 1))}\n")
        with open(os.path.join(directories["hyperopt_august2013_mod"],
                  "__init__.py"), "w") as fh:
            pass

        # Random search
        with open(os.path.join(directories["random_hyperopt_august2013_mod"],
                  "space.py"), "w") as fh:
            fh.write("from hyperopt import hp\n"
                     "from hyperopt.pyll import scope\n\n"
                     "space = {'C': scope.int(hp.quniform('C', -5, 15, 1)),\n'"
                     "gamma': scope.int(hp.quniform('gamma', -15, 3, 1))}\n")
        with open(os.path.join(directories["random_hyperopt_august2013_mod"],
                  "__init__.py"), "w") as fh:
            pass

        Spearmint
        with open(os.path.join(directories["spearmint_april2013_mod"], "config.pb"),
                  "w") as fh:
            fh.write('language: PYTHON\nname:     "HPOlib.cv"\nvariable {\n' \
                     ' name: "C"\n type: INT\n size: 1\n min:  -5\n' \
                     ' max:  15\n}\n\nvariable {\n name: "gamma"\n type: INT\n' \
                     ' size: 1\n min:  -15\n max: 3\n}')

        ########################################################################
        # Write the config

        # Find the ground truth
        glob_string = ground_truth_dir + dataset + "fold%d/" % fold + "gridsearch_1*"
        glob_results = glob.glob(glob_string)
        if len(glob_results) != 1:
            raise Exception("There must be only one gridsearch directory for "
                            "dataset %s and fold %d: %s" % (dataset, fold,
                                                            str(glob_results)))
        ground_truth = os.path.join(glob_results[0], "gridsearch.pkl")

        with open(os.path.join(experiment_dir, "config.cfg"), "w") as fh:
            content = "[HPOLIB]\n"
            content += "function = python -m pyMetaLearn.target_algorithm.reuse_results\n"
            content += "number_of_jobs = 50\n"
            content += "result_on_terminate = 1\n"
            content += "number_cv = 1\n"
            content += "[EXPERIMENT]\n"
            content += "test_fold = %s\n" % fold
            content += "test_folds = 3\n"
            content += "dataset = OPENML:%s\n" % dataset
            content += "dataset_name = %s\n" % dataset.split("_")[1]
            content += "ground_truth = %s\n" % ground_truth
            content += "[METALEARNING]\n"
            content += "distance_measure = l1\n"
            content += "datasets = " \
                       "/home/feurerm/thesis/experiments/2014_03_18_metaexperiments/datasets.txt\n"
            content += "experiments = " \
                       "/home/feurerm/thesis/experiments/2014_03_18_metaexperiments/experiments.txt\n"
            content += "\n"
            fh.write(content)

########################################################################
# Write the datasets and experiments file...
with open(os.path.join(experiments_directory, "datasets.txt"),
          "w") as fh:
    for ds in dataset_list:
        fh.write("OPENML:%s\n" % ds)

with open(os.path.join(experiments_directory, "experiments.txt"),
          "w") as fh:
    for ds in dataset_list:
        glob_string = ground_truth_dir + ds + "fold%d/" % fold + \
                      "gridsearch_1*"
        glob_results = glob.glob(glob_string)
        if len(glob_results) != 1:
            raise Exception("There must be only one gridsearch directory for "
                "dataset %s and fold %d: %s" % (dataset, fold, str(glob_results)))
        exp_pkl = os.path.join(glob_results[0], "gridsearch.pkl")
        fh.write("%s\n" % exp_pkl)
