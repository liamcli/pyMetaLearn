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

import cPickle
import logging
import os
import sys

import numpy as np

import HPOlib.wrapping_util


__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"


logger = logging.getLogger("HPOlib.spearmint_gitfork_mod")


#path_to_optimizer = "optimizers/spearmint_march2014_mod/"
version_info = ("# %76s #\n" % "git version march 2014")


def check_dependencies():
    try:
        import google.protobuf
        try:
            from google.protobuf.internal import enum_type_wrapper
        except ImportError:
            raise ImportError("Installed google.protobuf version is too old, you need at least 2.5.0")
    except ImportError:
        raise ImportError("Google protobuf cannot be imported. Are you sure "
                          "it's  installed?")
    try:
        import numpy
    except ImportError:
        raise ImportError("Numpy cannot be imported. Are you sure that it's"
                          " installed?")
    try:
        import scipy
    except ImportError:
        raise ImportError("Scipy cannot be imported. Are you sure that it's"
                          " installed?")


def build_spearmint_call(config, options, optimizer_dir):
    os.environ['PYTHONPATH'] = os.path.join(config.get('SPEARMINT', 'path_to_optimizer'), 'spearmint') + \
                               os.pathsep + os.environ['PYTHONPATH']
    print os.environ['PYTHONPATH']
    call = 'python ' + os.path.join(os.path.dirname(__file__),
                                    'bootstrap_spearmint.py')
    call = ' '.join([call,
                     config.get("EXPERIMENT", "task_args_pkl"),
                     config.get("METALEARNING", "tasks"),
                     config.get("METALEARNING", "experiments"),
                     config.get("EXPERIMENT", "openml_data_dir"),
                     os.path.join(optimizer_dir, config.get('SPEARMINT', 'config')),
                     " --distance_measure " + config.get('METALEARNING', 'distance_measure'),
                     " --cwd " + optimizer_dir,
                     " --seed " + config.get("HPOLIB", "seed"),
                     '--driver=local',
                     '--max-concurrent', config.get('HPOLIB', 'number_of_concurrent_jobs'),
                     '--max-finished-jobs', config.get('SPEARMINT', 'max_finished_jobs'),
                     '--polling-time', config.get('SPEARMINT', 'spearmint_polling_time'),
                     '--grid-size', config.get('SPEARMINT', 'grid_size'),
                     '--method',  config.get('SPEARMINT', 'method'),
                     '--method-args ' + config.get('SPEARMINT', 'method_args') +
                        'num_startup_jobs=' + config.get('METALEARNING', "num_bootstrap_examples"),
                     '--grid-seed', str(options.seed)])

    if config.has_option('METALEARNING', 'distance_keep_features'):
        keep = config.get('METALEARNING', 'distance_keep_features')
        if keep:
            call = ' '.join([call, ' --distance_keep_features ' + keep])

    if config.has_option('METALEARNING', 'distance_learner_params'):
        params = config.get('METALEARNING', 'distance_learner_params')
        if params:
            call = ' '.join([call, '--distance_kwargs', params])

    if config.has_option('METALEARNING', 'metafeatures_subset'):
        mf_subset = config.get('METALEARNING', 'metafeatures_subset')
        call = ' '.join([call, ' --metafeatures_subset ' + mf_subset])

    if config.get('SPEARMINT', 'method') != "GPEIChooser" and \
            config.get('SPEARMINT', 'method') != "GPEIOptChooser":
        logger.warning('WARNING: This chooser might not work yet\n')
        call = ' '.join([call, config.get("SPEARMINT", 'method_args')])
    return call


#noinspection PyUnusedLocal
def restore(config, optimizer_dir, **kwargs):
    raise NotImplementedError("Restoring is not possible for this optimizer")


#noinspection PyUnusedLocal
def main(config, options, experiment_dir, experiment_directory_prefix, **kwargs):
    # config:           Loaded .cfg file
    # options:          Options containing seed, restore_dir,
    # experiment_dir:   Experiment directory/Benchmark_directory
    # **kwargs:         Nothing so far

    time_string = HPOlib.wrapping_util.get_time_string()
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

    # Build call
    cmd = build_spearmint_call(config, options, optimizer_dir)

    # Set up experiment directory
    if not os.path.exists(optimizer_dir):
        os.mkdir(optimizer_dir)
        # Make a link to the Protocol-Buffer config file
        configpb = config.get('SPEARMINT', 'config')
        if not os.path.exists(os.path.join(optimizer_dir, configpb)):
            os.symlink(os.path.join(experiment_dir, optimizer_str, configpb),
                       os.path.join(optimizer_dir, configpb))

    path_to_optimizer = config.get("SPEARMINT", "path_to_optimizer")
    logger.info("### INFORMATION ################################################################")
    logger.info("# You're running %40s                      #" % path_to_optimizer)
    logger.info("%s" % version_info)
    logger.info("# A newer version might be available, but not yet built in.                    #")
    logger.info("# Please use this version only to reproduce our results on automl.org          #")
    logger.info("################################################################################")
    return cmd, optimizer_dir