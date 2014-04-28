import cPickle
import glob
import os

import pyMetaLearn.openml.manage_openml_data
from pyMetaLearn.openml.openml_task import OpenMLTask as Task

ground_truth_dir = "/home/feurerm/thesis/experiments/2014_04_17" \
    "_gather_new_metadata_from_openml/"
experiments_directory = "/home/feurerm/thesis/experiments/2014_04_23" \
    "_simple_metalearning/"
pyMetaLearn.openml.manage_openml_data.set_local_directory(
    "/home/feurerm/thesis/datasets/openml/")
local_directory = pyMetaLearn.openml.manage_openml_data.get_local_directory()
dataset_dir = os.path.join(local_directory, "datasets")

commands = []
metalearning_commands = []
bootstrap_commands = []

dataset_list = glob.glob(os.path.join(dataset_dir, "did*.xml"))
used_datasets = []
print dataset_list

optimizers = ["metalearning_optimizer",
              "spearmint_bootstrapped",
              "smac_2_06_01-dev",
              "hyperopt_august2013_mod",
              "spearmint_gitfork_mod",
              "random_hyperopt_august2013_mod"]

optimizer_locations = {"metalearning_optimizer": "/home/feurerm/thesis/Software/pyMetaLearn/optimizers/metalearn_optimizer/",
                       "spearmint_bootstrapped": "/home/feurerm/thesis/Software/pyMetaLearn/optimizers/spearmint_bootstrapped",
                       "smac_2_06_01-dev": "/home/feurerm/HPOlib/Software/HPOlib/optimizers/smac/",
                       "hyperopt_august2013_mod": "/home/feurerm/HPOlib/Software/HPOlib/optimizers/tpe/hyperopt",
                       "spearmint_gitfork_mod": "/home/feurerm/HPOlib/Software/HPOlib/optimizers/spearmint/spearmint_gitfork_mod",
                       "random_hyperopt_august2013_mod": "/home/feurerm/HPOlib/Software/HPOlib/optimizers/tpe/random"}

for dataset_filename in dataset_list:
    did = os.path.split(dataset_filename)[1]
    did = int(did.replace("did", "").replace(".xml", ""))

    task_file = os.path.join(pyMetaLearn
                .openml.manage_openml_data.get_local_directory(),
                "custom_tasks", "did_%d.pkl" % did)
    if not os.path.exists(task_file):
        print "Skipping dataset %s" % dataset_filename
        continue
    used_datasets.append(did)

    for fold in range(3):
        experiment_dir = os.path.join(experiments_directory, "did_%d_fold%s" % (did, fold))

        # Allows me to run this script again, adding new datasets
        if os.path.exists(experiment_dir):
            print "Skipping %s" % experiment_dir
            continue

        print experiment_dir
        os.mkdir(experiment_dir)

        ########################################################################
        # Pre-calculate the meta-features for this task...
        with open(task_file) as fh:
            task_args = cPickle.load(fh)
        task = Task(**task_args)
        dataset = pyMetaLearn.openml.manage_openml_data.get_local_dataset(
            task.dataset_id)
        X, Y = task.get_dataset()
        train_splits, test_splits = task._get_fold(X, Y, fold=fold, folds=3)
        dataset.get_metafeatures(subset_indices=tuple(train_splits))

        # Uncomment this to also calculate the metafeatures of the underlying
        #  10fold CV:
        # WARNING: this can take very long
        for inner_fold in range(10):
            inner_train, inner_test = task._get_fold(X[train_splits],
                Y[train_splits], fold=inner_fold, folds=10)
            dataset.get_metafeatures(subset_indices=tuple(inner_train))

        ########################################################################
        # Create all directories for the search spaces
        directories = dict()
        for optimizer in optimizers:
            directories[optimizer] = os.path.join(experiment_dir, optimizer)
            os.mkdir(directories[optimizer])

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

        # Bootstrapped spearmint is an instantiation of spearmint_gitfork_mod
        # Spearmint
        with open(os.path.join(directories["spearmint_gitfork_mod"], "config.pb"),
                  "w") as fh:
            fh.write('language: PYTHON\nname:     "HPOlib.cv"\nvariable {\n' \
                     ' name: "C"\n type: INT\n size: 1\n min:  -5\n' \
                     ' max:  15\n}\n\nvariable {\n name: "gamma"\n type: INT\n' \
                     ' size: 1\n min:  -15\n max: 3\n}')

        with open(os.path.join(directories["spearmint_bootstrapped"], "config.pb"),
                  "w") as fh:
            fh.write('language: PYTHON\nname:     "HPOlib.cv"\nvariable {\n' \
                     ' name: "C"\n type: INT\n size: 1\n min:  -5\n' \
                     ' max:  15\n}\n\nvariable {\n name: "gamma"\n type: INT\n' \
                     ' size: 1\n min:  -15\n max: 3\n}')

        ########################################################################
        # Write the config

        # Find the ground truth
        glob_string = ground_truth_dir + "did_%d_fold%s/" % (did, fold) + \
                      "gridsearch_1*"
        glob_results = glob.glob(glob_string)
        if len(glob_results) != 1:
            raise Exception("There must be only one gridsearch directory for "
                            "dataset %s and fold %d: %s" % (dataset, fold,
                                                            str(glob_results)))
        ground_truth = os.path.join(glob_results[0], "gridsearch.pkl")

        with open(os.path.join(experiment_dir, "config.cfg"), "w") as fh:
            content = "[HPOLIB]\n"
            content += "function = python -m pyMetaLearn.target_algorithm.reuse_results\n"
            content += "number_of_jobs = 200\n"
            content += "result_on_terminate = 1\n"
            content += "number_cv = 1\n"
            content += "[EXPERIMENT]\n"
            content += "test_fold = %s\n" % fold
            content += "test_folds = 3\n"
            content += "ground_truth = %s\n" % ground_truth
            content += "task_args_pkl = %s\n" % task_file
            content += "openml_data_dir = %s\n" % local_directory
            content += "[METALEARNING]\n"
            content += "distance_measure = l1\n"
            content += "metafeatures_subset = all\n"
            content += "datasets = " \
                       "/home/feurerm/thesis/experiments/2014_04_23_simple_metalearning/datasets.txt\n"
            content += "experiments = " \
                       "/home/feurerm/thesis/experiments/2014_04_23_simple_metalearning/experiments_fold%d.txt\n" % fold
            content += "\n"
            fh.write(content)

            # TODO: add a task-args pkl

        ########################################################################
        # Prepare SGE commands for a file

        for optimizer in optimizers:
            if optimizer not in ("metalearning_optimizer",
                                 "spearmint_gitfork_mod",
                                 "spearmint_bootstrapped"):
                for seed in range(1000, 10001, 1000):
                    commands.append("HPOlib-run -o %s -s %d --cwd %s\n"
                      % (optimizer_locations[optimizer], seed, experiment_dir))

            elif optimizer == "spearmint_gitfork_mod":
                for seed in range(1000, 10001, 1000):
                    commands.append("HPOlib-run -o %s -s %d "
                        "--SPEARMINT:path_to_optimizer %s --cwd %s\n"
                        % (optimizer_locations[optimizer], seed,
                         "/home/feurerm/thesis/Software/spearmint",
                         experiment_dir))

            elif optimizer == "spearmint_bootstrapped":
                for seed in range(1000, 10001, 1000):
                    bootstrap_commands.append("HPOlib-run -o %s -s %d "
                        "--SPEARMINT:path_to_optimizer %s --cwd %s"
                        % (optimizer_locations[optimizer], seed,
                         "/home/feurerm/thesis/Software/spearmint",
                         experiment_dir))

            elif optimizer == "metalearning_optimizer":   # Metalearner
                commands.append("HPOlib-run -o %s --cwd %s\n"
                     % (optimizer_locations[optimizer], experiment_dir))
                metalearning_commands.append("HPOlib-run -o %s --cwd %s"
                     % (optimizer_locations[optimizer], experiment_dir))

            else:
                raise ValueError("Unknown optimizer %s" % optimizer)

################################################################################
# Write the actual SGE files
test_commands = []

commands.sort()
smbo_on_grid_file = os.path.join(experiments_directory, "smbo_on_grid.txt")
with open(smbo_on_grid_file, "a") as fh:
    for command in commands:
        fh.write(command)

# Add one command per optimizer to the test commands
step_size = len(commands) / (len(optimizers) - 2)
command_indices = [i * step_size for i in range((len(optimizers) - 1))]
for command_idx in command_indices:
    test_commands.append(commands[command_idx])

metalearning_commands.sort()
instantiations = {"all": "--METALEARNING:metafeatures_subset all",
                  "pfahringer_2000_experiment1": "--METALEARNING:metafeatures_subset pfahringer_2000_experiment1",
                  # "pfahringer_2": "--METALEARNING:metafeatures_subset pfahringer_2000_experiment2",
                  "yogotama_2014": "--METALEARNING:metafeatures_subset yogotama_2014",
                  "bardenet_2013_boost": "--METALEARNING:metafeatures_subset bardenet_2013_boost",
                  "bardenet_2013_nn": "--METALEARNING:metafeatures_subset bardenet_2013_nn",}
distance_measures = ["l1", "l2"]

for instance in instantiations:
    for measure in distance_measures:
        metal_on_grid_file = os.path.join(experiments_directory,
            "metal_on_grid_%s-distance_%s-features.txt" % (measure, instance))
        with open(metal_on_grid_file, "a") as fh:
            for command in metalearning_commands:
                command = "%s --METALEARNING:distance_measure %s  " \
                         "--METALEARNING:metafeatures_subset %s\n" % \
                         (command, measure, instance)
                fh.write(command)
            test_commands.append(command)



bootstrap_commands.sort()
bootstrap_samples = (2, 5, 10)
for instance in instantiations:
    for measure in distance_measures:
        for samples in bootstrap_samples:
            bootstrap_on_grid_file = os.path.join(experiments_directory,
                "bootstrap_on_grid_%dsamples_%s-dist_%s-metafeatures.txt" %
                (samples, measure, instance))
            with open(bootstrap_on_grid_file, "a") as fh:
                for command in bootstrap_commands:
                    command = "%s --METALEARNING:metafeatures_subset %s " \
                             "--METALEARNING:distance_measure %s " \
                             "--METALEARNING:num_bootstrap_examples " \
                             "%d --SPEARMINT:method_args num_startup_jobs=%d\n"\
                             % (command, instance, measure, samples, samples)
                    fh.write(command)
                test_commands.append(command)

# Also write a file with test commands - one command per kind of configuration
test_commands_file = os.path.join(experiments_directory, "test_commands.txt")
if not os.path.exists(test_commands_file):
    with open(test_commands_file, "w") as fh:
        for command in test_commands:
            fh.write(command)

with open(os.path.join(experiments_directory, "init_experiment.sh"), "w") as fh:
    fh.write("source /home/feurerm/.bashrc\n")
    fh.write("source /home/feurerm/thesis/virtualenvs/hpolib_metagpu"
             "/set_path.sh\n")
    fh.write("export PYTHONPATH=$PYTHONPATH:/home/feurerm/thesis/Software\n")

########################################################################
# Write the datasets and experiments file...
with open(os.path.join(experiments_directory, "datasets.txt"),
          "w") as fh:
    for ds in used_datasets:
        fh.write("OPENML:%d\n" % ds)

for fold in range(3):
    with open(os.path.join(experiments_directory, "experiments_fold%d.txt" % fold),
              "w") as fh:
        for ds in used_datasets:
            glob_string = ground_truth_dir + "did_%d_fold%d/" % (ds, fold) + \
                          "gridsearch_1*"
            glob_results = glob.glob(glob_string)
            if len(glob_results) != 1:
                raise Exception("There must be only one gridsearch directory for "
                    "dataset %s and fold %d: %s" % (dataset, fold, str(glob_results)))
            exp_pkl = os.path.join(glob_results[0], "gridsearch.pkl")
            fh.write("%s\n" % exp_pkl)
