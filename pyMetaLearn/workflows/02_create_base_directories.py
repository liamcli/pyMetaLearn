import glob
import os

import pyMetaLearn.openml.manage_openml_data

experiments_directory = \
    "/home/feurerm/thesis/experiments/2014_04_17_gather_new_metadata_from_openml"
pyMetaLearn.openml.manage_openml_data.set_local_directory(
    "/home/feurerm/thesis/datasets/openml/")
local_directory = pyMetaLearn.openml.manage_openml_data.get_local_directory()
dataset_dir = os.path.join(local_directory, "datasets")
commands = []

optimizers = ["gridsearch", "smac_2_06_01-dev",
              "hyperopt_august2013_mod", "spearmint_april2013_mod",
              "random_hyperopt_august2013_mod"]

optimizer_locations = {"gridsearch": "/home/feurerm/HPOlib/working_directory/gridsearch/",
                       "smac_2_06_01-dev": "/home/feurerm/HPOlib/Software/HPOlib/optimizers/smac/",
                       "hyperopt_august2013_mod": "/home/feurerm/HPOlib/Software/HPOlib/optimizers/tpe/hyperopt",
                       "spearmint_april2013_mod": "/home/feurerm/HPOlib/Software/HPOlib/optimizers/spearmint/spearmint_april2013",
                       "random_hyperopt_august2013_mod": "/home/feurerm/HPOlib/Software/HPOlib/optimizers/tpe/random"}

dataset_list = glob.glob(os.path.join(dataset_dir, "did*xml"))

for dataset in dataset_list:
    did = os.path.split(dataset)[1]
    did = int(did.replace("did", "").replace(".xml", ""))

    task_file = os.path.join(pyMetaLearn
                .openml.manage_openml_data.get_local_directory(),
                "custom_tasks", "did_%d.pkl" % did)
    if not os.path.exists(task_file):
        print "Skipping dataset %s" % dataset
        continue

    for fold in range(3):
        experiment_dir = os.path.join(experiments_directory,
                                      "did_%d_fold%s" % (did, fold))

        # Allows me to run this script again, adding new datasets
        if os.path.exists(experiment_dir):
            print "Skipping %s" % experiment_dir
            continue

        print experiment_dir
        os.mkdir(experiment_dir)

        ########################################################################
        # Create all directories for the search spaces
        directories = dict()

        for optimizer in optimizers:
            directories[optimizer] = os.path.join(experiment_dir, optimizer)
            os.mkdir(directories[optimizer])

        # Gridsearch
        with open(os.path.join(directories["gridsearch"], "params.pcs"),
                  "w") as fh:
            fh.write("C {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "
                     "10, 11, 12, 13, 14, 15}\n"
                     "gamma {-15, -14, -13, -12, -11, -10, -9, -8, -7, -6, "
                     "-5, -4, -3, -2, -1, 0, 1, 2, 3}")

        # SMAC
        with open(os.path.join(directories["smac_2_06_01-dev"], "params.pcs"),
                  "w") as fh:
            fh.write("C [-5, 15] [1]\ngamma [-15, 3] [1]")

        # Hyperopt
        with open(os.path.join(directories["hyperopt_august2013_mod"],
                  "space.py"), "w") as fh:
            fh.write("from hyperopt import hp\n"
                     "from hyperopt.pyll import scope\n\n"
                     "space = {'C': scope.int(hp.uniform('C', -5, 15)),\n'"
                     "gamma': scope.int(hp.uniform('gamma', -15, 3))}\n")
        with open(os.path.join(directories["hyperopt_august2013_mod"],
                  "__init__.py"), "w") as fh:
            pass

        # Random search
        with open(os.path.join(directories["random_hyperopt_august2013_mod"],
                  "space.py"), "w") as fh:
            fh.write("from hyperopt import hp\n"
                     "from hyperopt.pyll import scope\n\n"
                     "space = {'C': scope.int(hp.uniform('C', -5, 15)),\n'"
                     "gamma': scope.int(hp.uniform('gamma', -15, 3))}\n")
        with open(os.path.join(directories["random_hyperopt_august2013_mod"],
                  "__init__.py"), "w") as fh:
            pass

        # Spearmint
        with open(os.path.join(directories["spearmint_april2013_mod"], "config.pb"),
                  "w") as fh:
            fh.write('language: PYTHON\nname:     "HPOlib.cv"\nvariable {\n' \
                     ' name: "C"\n type: FLOAT\n size: 1\n min:  -5\n' \
                     ' max:  15\n}\n\nvariable {\n name: "gamma"\n type: FLOAT\n' \
                     ' size: 1\n min:  -15\n max: 3\n}')

        ########################################################################
        # Write the config

        with open(os.path.join(experiment_dir, "config.cfg"), "w") as fh:
            content = "[HPOLIB]\n"
            content += "function = python -m pyMetaLearn.target_algorithm.libsvm\n"
            content += "number_of_jobs = 399\n"
            content += "result_on_terminate = 1\n"
            content += "number_cv_folds = 10\n"
            content += "[EXPERIMENT]\n"
            content += "test_fold = %d\n" % fold
            content += "task_args_pkl = %s\n" % task_file
            content += "openml_data_dir = %s\n" % local_directory
            content += "\n"
            fh.write(content)

        ########################################################################
        # Prepare SGE commands for a file

        for optimizer in optimizers:
            if optimizer != "gridsearch":
                for seed in range(1000, 10001, 1000):
                    commands.append("HPOlib-run -o %s -s %d --cwd %s\n"
                      % (optimizer_locations[optimizer], seed, experiment_dir))
            else:
                commands.append("HPOlib-run -o %s --cwd %s\n"
                     % (optimizer_locations[optimizer], experiment_dir))

commands.sort()
sge_commands_file = os.path.join(experiments_directory, "sge_commands.txt")
with open(sge_commands_file, "a") as fh:
    for command in commands:
        fh.write(command)


with open(os.path.join(experiments_directory, "init_experiment.sh"), "w") as fh:
    fh.write("source /home/feurerm/.bashrc\n")
    fh.write("source /home/feurerm/thesis/virtualenvs/hpolib_metagpu"
             "/set_path.sh\n")
    fh.write("export PYTHONPATH=$PYTHONPATH:/home/feurerm/thesis/Software\n")