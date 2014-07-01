##
# wrapping: A program making it easy to use hyperparameter
# optimization software.
# Copyright (C) 2013 Katharina Eggensperger and Matthias Feurer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import itertools
import logging
import os
import StringIO
import subprocess

import numpy as np

import HPOlib.wrapping_util as wrapping_util
import HPOlib.Experiment as Experiment
import HPOlib.dispatcher.dispatcher as dispatcher

import pyMetaLearn.optimizers.metalearn_optimizer.metalearner as metalearner


logger = logging.getLogger("HPOlib.smac_2_06_01-warmstart")


__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"

version_info = ["Automatic Configurator Library ==> v2.06.01-development-643 (a1f71813a262)",
                "Random Forest Library ==> v1.05.01-development-95 (4a8077e95b21)",
                "SMAC ==> v2.06.01-development-620 (9380d2c6bab9)"]

#optimizer_str = "smac_2_06_01-dev"


def check_dependencies():
    process = subprocess.Popen("which java", stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, shell=True, executable="/bin/bash")
    stdoutdata, stderrdata = process.communicate()

    if stdoutdata is not None and "java" in stdoutdata:
        pass
    else:
        raise Exception("Java cannot not be found. "
                        "Are you sure that it's installed?\n"
                        "Your $PATH is: " + os.environ['PATH'])


def build_smac_call(config, options, optimizer_dir):
    import HPOlib
    algo_exec_dir = os.path.dirname(HPOlib.__file__)

    call = config.get('SMAC', 'path_to_optimizer') + "/smac"
    call = " ".join([call, '--numRun', str(options.seed),
                    '--scenario-file', os.path.join(optimizer_dir, 'scenario.txt'),
                    '--cutoffTime', config.get('SMAC', 'cutoff_time'),
                    # The instance file does interfere with state restoration, it will only
                    # be loaded if no state is restored (look further down in the code
                    # '--instanceFile', config.get('SMAC', 'instanceFile'),
                    '--intraInstanceObj', config.get('SMAC', 'intra_instance_obj'),
                    '--runObj', config.get('SMAC', 'run_obj'),
                    # '--testInstanceFile', config.get('SMAC', 'testInstanceFile'),
                    '--algoExec',  '"python', os.path.join(algo_exec_dir,
                        'dispatcher', 'dispatcher.py') + '"',
                    #                config.get('SMAC', 'algo_exec')) + '"',
                    '--execDir', optimizer_dir,
                    '-p', os.path.join(optimizer_dir, os.path.basename(config.get('SMAC', 'p'))),
                    # The experiment dir MUST not be specified when restarting, it is set
                    # further down in the code
                    # '--experimentDir', optimizer_dir,
                    '--numIterations', config.get('SMAC', 'num_iterations'),
                    '--totalNumRunsLimit', config.get('SMAC', 'total_num_runs_limit'),
                    '--outputDirectory', optimizer_dir,
                    '--numConcurrentAlgoExecs', config.get('SMAC', 'num_concurrent_algo_execs'),
                    # '--runGroupName', config.get('SMAC', 'runGroupName'),
                    '--maxIncumbentRuns', config.get('SMAC', 'max_incumbent_runs'),
                    '--retryTargetAlgorithmRunCount',
                    config.get('SMAC', 'retry_target_algorithm_run_count'),
                    '--save-runs-every-iteration true',
                    '--intensification-percentage',
                    config.get('SMAC', 'intensification_percentage'),
                    '--rf-split-min', config.get('SMAC', 'rf_split_min'),
                    '--validation', config.get('SMAC', 'validation')])

    if config.getboolean('SMAC', 'deterministic'):
        call = " ".join([call, '--deterministic true'])

    if config.getboolean('SMAC', 'adaptive_capping') and \
            config.get('SMAC', 'run_obj') == "RUNTIME":
        call = " ".join([call, '--adaptiveCapping true'])
    
    if config.getboolean('SMAC', 'rf_full_tree_bootstrap'):
        call = " ".join([call, '--rf-full-tree-bootstrap true'])

    call += " --restore-scenario %s" % optimizer_dir
    return call


def create_smac_files_(config, output_dir):
    context = metalearner.setup(None, config=config)
    meta_base = context['meta_base']
    experiment = meta_base.get_experiment(context['dataset_name'])
    hp_list = metalearner.metalearn_suggest_all(None, context)
    num_bootstrap_examples = config.getint("METALEARNING",
                                           "num_bootstrap_examples")

    runs_and_results = StringIO.StringIO()
    runs_and_results.write("Run Number,Run History Configuration ID,Instance ID,"
                           "Response Value (y),Censored?,Cutoff Time Used,Seed,"
                           "Runtime,Run Length,Run Result Code,Run Quality,SMAC"
                           " Iteration,SMAC Cumulative Runtime,Run Result,"
                           "Additional Algorithm Run Data,Wall Clock Time,\n")
    paramstrings = StringIO.StringIO()
    bootstrap_results = []

    experiment_as_dict = dict()
    for exp in experiment:
        experiment_as_dict[str(exp.params)] = exp.result

    for idx in range(num_bootstrap_examples):
        result = experiment_as_dict[str(hp_list[idx])]
        bootstrap_results.append(result)

        iteration = int(idx/2)
        string = "%s,%s,%s,%f,0,108000,-1,%f,1,1,%f,%d,%f,SAT,Aditional data," \
                 "%f" % (idx+1, idx+1, 1, result, 1.0,
                         result, iteration, float(idx+1), 1.0)
        runs_and_results.write(string + "\n")

    bootstrap_parameters = []
    # TODO: obtain original parameter names to convert back to LOG2 and other
    #  transformations...
    for idx in range(num_bootstrap_examples):
        # TODO: the params contain some default values which are important
        # because SMAC expects that the paramstrings file is sorted
        # alphabetically and contains all hyperparameters
        params_ = {'classifier': 'random_forest', 'liblinear:LOG2_C': '1',
                  'liblinear:loss': 'l2', 'liblinear:penalty': 'l1',
                  'libsvm_svc:LOG2_C': '1', 'libsvm_svc:LOG2_gamma': '1',
                  'pca:keep_variance': '0', 'preprocessing': 'None',
                  'random_forest:criterion': 'gini',
                  'random_forest:max_features': '10',
                  'random_forest:min_samples_split': '1'}

        params = {}

        # Remove all hyperparameters with a trailing minus
        for param in params_:
            value = hp_list[idx].get(param)
            if value is None:
                value = hp_list[idx].get("-" + param)
                if value is None:
                    value = params_[param]

            params[param] = value

        # dispatcher.py puts the data to the trials pickle after LOG stuff
        # etc is removed, thus the LOG2 etc does not have to be addad again
        bootstrap_parameters.append(params)

        param_list = []
        logger.info("%s" % str(params))
        for param in params:
            if param == "liblinear:C":
                key = "liblinear:LOG2_C"
                value = int(np.log2(params[param]))
            elif param == "libsvm_svc:C":
                key = "libsvm_svc:LOG2_C"
                value = int(np.log2(params[param]))
            elif param == "libsvm_svc:gamma":
                key = "libsvm_svc:LOG2_gamma"
                value = int(np.log2(params[param]))
            else:
                key = param
                value = params[param]

            param_list.append("%s='%s'" % (key, value))

        param_list.sort()
        paramstrings.write("%d: %s\n" % (idx+1, ", ".join(param_list)))

    with open(os.path.join(output_dir, "runs_and_results-it%d.csv" %
            iteration), "w") as fh:
        runs_and_results.seek(0)
        for line in runs_and_results:
            fh.write(line)
    with open(os.path.join(output_dir, "paramstrings-it%d.txt" % iteration),
              "w") as fh:
        paramstrings.seek(0)
        for line in paramstrings:
            fh.write(line)

    return bootstrap_parameters, bootstrap_results


#noinspection PyUnusedLocal
def main(config, options, experiment_dir, experiment_directory_prefix, **kwargs):
    # config:           Loaded .cfg file
    # options:          Options containing seed, restore_dir, 
    # experiment_dir:   Experiment directory/Benchmark_directory
    # **kwargs:         Nothing so far
    time_string = wrapping_util.get_time_string()

    optimizer_str = os.path.splitext(os.path.basename(__file__))[0]

    # Find experiment directory
    if options.restore:
        if not os.path.exists(options.restore):
            raise Exception("The restore directory does not exist")
        optimizer_dir = options.restore
    else:
        optimizer_dir = os.path.join(experiment_dir,
                                     experiment_directory_prefix
                                     + optimizer_str + "_" +
                                     str(options.seed) + "_" + time_string)

    # Set up experiment directory
    if not os.path.exists(optimizer_dir):
        os.mkdir(optimizer_dir)
        # TODO: This can cause huge problems when the files are located
        # somewhere else?
        space = config.get('SMAC', "p")
        abs_space = os.path.abspath(space)
        parent_space = os.path.join(experiment_dir, optimizer_str, space)
        if os.path.exists(abs_space):
            space = abs_space
        elif os.path.exists(parent_space):
            space = parent_space
        else:
            raise Exception("SMAC search space not found. Searched at %s and "
                            "%s" % (abs_space, parent_space))

        if not os.path.exists(os.path.join(optimizer_dir, os.path.basename(space))):
            os.symlink(os.path.join(experiment_dir, optimizer_str, space),
                       os.path.join(optimizer_dir, "param.pcs"))
        
        # Copy the smac search space and create the instance information
        fh = open(os.path.join(optimizer_dir, 'instances.txt'), "w")
        for i in range(config.getint('HPOLIB', 'number_cv_folds')):
            fh.write(str(i) + "\n")
        fh.close()
        
        fh = open(os.path.join(optimizer_dir, "scenario.txt"), "w")
        fh.close()

    bootstrap_parameters, bootstrap_results =\
        create_smac_files_(config, optimizer_dir)
    cmd = build_smac_call(config, options, optimizer_dir)

    # TODO: add stuff to the pickle
    trials = Experiment.Experiment(optimizer_dir,
        experiment_directory_prefix + "smac_warmstart",
        folds=config.getint('HPOLIB', 'number_cv_folds'),
        max_wallclock_time=config.get('HPOLIB', 'cpu_limit'))

    num_bootstrap_examples = config.getint("METALEARNING",
                                           "num_bootstrap_examples")
    for params, res in itertools.izip(bootstrap_parameters, bootstrap_results):
        trial_index = dispatcher.get_trial_index(trials, 0, params)
        trials.set_one_fold_running(trial_index, 0)
        trials.set_one_fold_complete(trial_index, 0, res, 1.0)
        trials._save_jobs()
    del trials

    logger.info("### INFORMATION ################################################################")
    logger.info("# You're running %40s                      #" % config.get('SMAC', 'path_to_optimizer'))
    for v in version_info:
        logger.info("# %76s #" % v)
    logger.info("# A newer version might be available, but not yet built in.                    #")
    logger.info("# Please use this version only to reproduce our results on automl.org          #")
    logger.info("################################################################################")
    return cmd, optimizer_dir