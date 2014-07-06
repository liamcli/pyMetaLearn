import ast
from argparse import ArgumentParser
from collections import OrderedDict
import cPickle
import os
import sys

import numpy as np

import HPOlib.wrapping_util as wrapping_util

def main(spearmint_path, experiment_dir):
    sys.path.append(spearmint_path)
    sys.path.append(os.path.join(spearmint_path, ".."))

    os.chdir(experiment_dir)
    ei_dir = os.path.join(os.getcwd(), "ei")
    import matplotlib.pyplot as plt

    try:
        import ExperimentGrid
    except ImportError as e:
        sys.stderr.write(e.message)
        raise ImportError("Could not import ExperimentGrid. "
                         "PYTHONPATH is %s" % str(sys.path))

    config = wrapping_util.load_experiment_config_file()
    ground_truth = config.get("EXPERIMENT", "ground_truth")
    with open(ground_truth) as fh:
        ground_truth = cPickle.load(fh)

    if os.path.exists(ei_dir) and os.path.isdir(ei_dir):
        expt_grid = ExperimentGrid.ExperimentGrid(".")
        variables = expt_grid.vmap.variables
        names = [(variable["name"], variable) for variable in variables]
        names.sort()
        variables_dict = OrderedDict(names)

        for ei_file in os.listdir(ei_dir):
            if not ".txt" in ei_file:
                continue

            ei_file = os.path.join(ei_dir, ei_file)
            print "Plotting", ei_file

            x_range = variables_dict.values()[0]["max"] - variables_dict \
                .values()[0]["min"]
            y_range = variables_dict.values()[1]["max"] - variables_dict \
                .values()[1]["min"]
            image = np.zeros((x_range + 1, y_range + 1), dtype=np.float64)
            values = image.copy()


            figure = plt.figure()
            ax = figure.add_subplot(122)
            ax.set_ylabel(variables_dict.keys()[0])
            ax.set_xlabel(variables_dict.keys()[1])
            ax.set_title("Grid Search")

            best = sys.maxint
            best_param_values = None
            best_params = []
            iteration_found = 0
            for i, trial in enumerate(ground_truth["trials"]):
                params = trial["params"]

                # Hack to remove all trailing - from the params which are
                # accidently in the experiment pickle of the current HPOlib version
                for key in params:
                    if key[0] == "-":
                        params[key[1:]] = params[key]
                        params.pop(key)

                param_0 = int(params[variables[0]["name"]]) -\
                          int(variables[0]["min"])
                param_1 = int(params[variables[1]["name"]]) -\
                          int(variables[1]["min"])

                x = param_0 if variables[0]["name"] ==\
                               variables_dict.keys()[0] else param_1
                y = param_0 if variables[1]["name"] ==\
                               variables_dict.keys()[0] else param_1

                values[x][y] = trial["result"]
                param_string = str("(%s: %d, %s %d)" % (variables[0]["name"],
                    int(params[variables[0]["name"]]),
                    variables[1]["name"],
                    int(params[variables[1]["name"]])))

                if trial["result"] < best:
                    best_param_values = (x, y)
                    best = trial["result"]
                    best_params = [param_string]
                elif trial["result"] == best:
                    best_params.append(param_string)

            ax.matshow(values, cmap=plt.cm.Blues, interpolation='nearest',
                        origin='lower')
            ax.scatter(best_param_values[0], best_param_values[1],
                       color="r")

            ax = figure.add_subplot(121)
            fh = open(ei_file)
            ei_values = ast.literal_eval(fh.read())

            found_after = sys.maxint
            for i, grid_idx in enumerate(ei_values):
                params = expt_grid.get_params(grid_idx)
                param_0 = params[0].int_val[0] - int(variables[0]["min"])
                param_1 = params[1].int_val[0] - int(variables[1]["min"])

                x = param_0 if variables[0]["name"] == variables_dict.keys()[
                    0] else param_1
                y = param_0 if variables[1]["name"] == variables_dict.keys()[
                    0] else param_1
                image[x][y] = ei_values[grid_idx]

            ax.matshow(image, cmap=plt.cm.Blues, interpolation='nearest',
                        origin='lower')
            ax.scatter(best_param_values[0], best_param_values[1],
                       color="r")

            #x_ticks = ax.get_xticks().tolist()
            #y_ticks = ax.get_yticks().tolist()
            #for i in range(len(x_ticks)):
            #    x_ticks[i] = str(int(x_ticks[i]) - int(variables[1]["min"]))
            #for i in range(len(y_ticks)):
            #    y_ticks[i] = str(int(y_ticks[i]) - int(variables[0]["min"]))
            #ax.set_xticklabels(x_ticks)
            #ax.set_yticklabels(y_ticks)

            ax.set_title("Expected Improvement")
            ax.set_ylabel(variables_dict.keys()[0])
            ax.set_xlabel(variables_dict.keys()[1])

            if type(best_params) == list:
                best_params = ", ".join(best_params)
            plt.figtext(0.5, 0.965, "Best value is %s found at "
                        "%s" % (best, best_params),
                        ha='center', color='black')
            plt.savefig(os.path.join(ei_dir, ei_file.replace(".txt", ".png")))
            plt.close()
            fh.close()

    else:
        print "No ei directory found in %s" % os.getcwd()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--spearmint", type=str, required=True,
        help="Path to the spearmint library. Needed in order to import "
             "spearmint.ExperimentGrid")
    parser.add_argument("--dir", type=str, required=True,
        help="Experiment directory in which the script searches for a "
             "directory with EI values.")
    args = parser.parse_args()
    main(args.spearmint, args.dir)