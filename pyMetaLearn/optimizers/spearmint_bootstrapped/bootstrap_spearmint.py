import argparse
import ast
from collections import OrderedDict
import os
import sys

import HPOlib.wrapping_util as wrapping_util
import pyMetaLearn.optimizers.metalearn_optimizer.metalearner as ml
import pyMetaLearn.openml.manage_openml_data


def start_spearmint(task_file, task_filenames, experiment_filenames,
                    spearmint_arguments, cwd, distance='l1',
                    seed=None, use_features='', distance_kwargs=None, subset='all'):
    print os.getcwd()
    config = wrapping_util.load_experiment_config_file()

    metalearner = ml.MetaLearningOptimizer(task_file, task_filenames,
                                           experiment_filenames, cwd,
                                           distance, seed, use_features,
                                           distance_kwargs, subset)

    hp_list = metalearner.metalearning_suggest_all()
    # TODO remove the config from here!
    num_bootstrap_examples = config.getint("METALEARNING",
                                         "num_bootstrap_examples")

    path_to_spearmint = config.get("SPEARMINT", "path_to_optimizer")
    import spearmint.main as spearmint

    for i, params in enumerate(hp_list[:num_bootstrap_examples]):
        fixed_params = OrderedDict()
        # Hack to remove all trailing - from the params which are
        # accidently in the experiment pickle of the current HPOlib version
        for key in params:
            if key[0] == "-":
                fixed_params[key[1:]] = params[key]
            else:
                fixed_params[key] = params[key]

        hp_list[i] = fixed_params

    sys.stderr.write("Initialize spearmint with " + str(hp_list[:num_bootstrap_examples]))
    sys.stderr.write("\n")
    sys.stderr.write(str(subset) + "\n")
    sys.stderr.flush()

    spearmint.main(args=spearmint_arguments, pre_eval=hp_list[:num_bootstrap_examples])


def parse_parameters(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("task_file", type=str,
                        help="The task which should be optimized.")
    parser.add_argument("task_files_list", type=str,
                        help="A list with all task files for which "
                             "should be considered for metalearning")
    parser.add_argument("experiment_files_list", type=str,
                        help="A list with all experiment pickles which "
                             "should be considered for metalearning")
    parser.add_argument("metalearning_directory", type=str,
                        help="A directory with the metalearning datastructure")
    parser.add_argument("-d", "--distance_measure", type=str, default='l1',
                        choices=['l1', 'l2', 'learned', 'random', 'mfs_l1',
                                 'mfw_l1'])
    parser.add_argument("--metafeatures_subset", type=str, default='all',
                        choices=["pfahringer_2000_experiment1",
                                 "all", "yogotama_2014",
                                 "bardenet_2013_boost", "bardenet_2013_nn"])
    parser.add_argument("--distance_keep_features", type=str, default='',)
    parser.add_argument("--distance_kwargs", type=str, default='')
    parser.add_argument("--cli_target")
    # parser.add_argument("-p", "--params", required=True)
    parser.add_argument("--cwd", type=str)
    parser.add_argument("-s", "--seed", type=int, default=1)
    args, unknown = parser.parse_known_args()
    return args, unknown


def main():
    args, unknown = parse_parameters()
    if args.cwd:
        os.chdir(args.cwd)

    # TODO check if the directory contains a valid directory structure!
    # No, don't as we're inside an experiment directory; check if the openml
    # directory contains everything which is necessary!
    pyMetaLearn.openml.manage_openml_data.set_local_directory(args.metalearning_directory)

    with open(args.task_files_list) as fh:
         task_filenames = fh.readlines()
    with open(args.experiment_files_list) as fh:
        experiment_filenames = fh.readlines()

    if args.distance_kwargs:
        distance_kwargs = ast.literal_eval(args.distance_kwargs)
    else:
        distance_kwargs = None

    start_spearmint(args.task_file, task_filenames,
        experiment_filenames, unknown, args.cwd, distance=args.distance_measure,
        seed=args.seed, use_features=args.distance_keep_features,
        distance_kwargs=distance_kwargs, subset=args.metafeatures_subset)


if __name__ == "__main__":
    main()
    exit(0)
