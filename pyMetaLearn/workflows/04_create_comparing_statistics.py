from argparse import ArgumentParser
from collections import OrderedDict
import cPickle
import glob
import itertools
import numpy as np
import os
import re
import scipy.stats
import sys

from matplotlib import pyplot as plt

import HPOlib.Plotting.plot_util as plot_util
import HPOlib.Plotting.generateTexTable as generate_tex_table
import HPOlib.Plotting.plotTraceWithStd_perEval as plotTraceWithStd_perEval
import HPOlib.Plotting.statistics as statistics

import pyMetaLearn.openml.manage_openml_data
import pyMetaLearn.metafeatures.metafeatures as mf_module

ground_truth_dir = "/home/feurerm/thesis/experiments/2014_04_17" \
    "_gather_new_metadata_from_openml/"
experiments_directory = "/home/feurerm/thesis/experiments/2014_04_23" \
    "_simple_metalearning/"
plot_dir = os.path.join(experiments_directory, "plots")
pyMetaLearn.openml.manage_openml_data.set_local_directory(
    "/home/feurerm/thesis/datasets/openml/")
local_directory = pyMetaLearn.openml.manage_openml_data.get_local_directory()
openml_dataset_dir = os.path.join(local_directory, "datasets")


header_template = """\\begin{table}[t]
\\begin{tabularx}{\\textwidth}{l{%- for name in result_values -%}|Xr{%- endfor -%}}
\\toprule
\multicolumn{1}{l}{}
{%- for name in result_values -%}
&\multicolumn{2}{c}{\\bf {{name}}}
{%- endfor -%}
\\\\
\multicolumn{1}{r}{\\#evals}
{%- for name in result_values -%}
&\\multicolumn{1}{l}{Valid.\\ loss} &\\multicolumn{1}{r}{Best loss}
{%- endfor -%}
\\\\
\\toprule
"""

line_template = """{{ evals }}
{%- for name in result_values -%}
{%- set results = result_values[name] -%}
{{ ' & ' }}{% if results['mean_best'] == True %}\\textbf{ {%- endif %}{{results['mean']|round(3, 'floor') }}{% if results['mean_best'] == True %}}{% endif %}$\\pm${{ results['std']|round(3, 'floor')}} & {{results['min']|round(3, 'floor') }}{%- endfor %} \\\\
"""

tex_header = """\documentclass{article} % For LaTeX2

\usepackage[a2paper, left=5mm, right=5mm, top=5mm, bottom=5mm]{geometry}
\usepackage{multirow}           % import command \multicolmun
\usepackage{tabularx}           % Convenient table formatting
\usepackage{booktabs}           % provides 	oprule, \midrule andottomrule
\usepackage{array}              % needed for rotated text
\usepackage{graphicx}           % needed for rotated text

\\begin{document}

"""

fh = open("%s/giant_tex_table_py.tex"  % plot_dir, "w")
fh.write(tex_header)

parser = ArgumentParser()
args, unknown = parser.parse_known_args()

try:
    os.mkdir(plot_dir)
except:
    pass

dataset_list = glob.glob(os.path.join(openml_dataset_dir, "did*.xml"))
datasets = []
for dataset_filename in dataset_list:
    did = os.path.split(dataset_filename)[1]
    did = int(did.replace("did", "").replace(".xml", ""))

    task_file = os.path.join(pyMetaLearn
                .openml.manage_openml_data.get_local_directory(),
                "custom_tasks", "did_%d.pkl" % did)
    if not os.path.exists(task_file):
        print "Skipping dataset %s" % dataset_filename
        continue
    datasets.append(did)
datasets = datasets

print datasets, len(datasets)

datasets.sort()
optimizers = OrderedDict([#("Meta\_VSM",
                  #"%s/metalearn_optimizer_10000_*/metalearn_optimizer.pkl"),
                  #       ("Meta\_L1",
                  #"%s/metalearn_optimizer_10001_*/metalearn_optimizer.pkl"),
                  #       ("Meta\_L2",
                  #"%s/metalearn_optimizer_10002_*/metalearn_optimizer.pkl"),
                         ("SMAC", "%s/smac_2_06_01-dev_*/smac_2_06_01-dev"
                                  ".pkl"),
           #              ("Spearmint",
           #       "%s/spearmint_april2013_mod_*/spearmint_april2013_mod.pkl"),
                         #("Spearmint/Meta10",
            #"%s/spearmint_gitfork_mod_bootstrap10_*/spearmint_gitfork_mod
            # .pkl"),
                         ("random", "%s/random_hyperopt_august2013_mod"
                        "*/random_hyperopt_august2013_mod.pkl"),
                         ("TPE",
    "%s/hyperopt_august2013_mod_*/hyperopt_august2013_mod" \
                     ".pkl"),
                         ("Spearmint/Grid",
            "%s/spearmint_gitfork_mod_*/spearmint_gitfork_mod.pkl"),
                         ("Spearmint/Learned2",
            "%s/bootstrapped2_learnedspearmint_gitfork_mod_*/*spearmint_gitfork_mod"
            ".pkl"), ("Spearmint/Learned5",
            "%s/bootstrapped5_learnedspearmint_gitfork_mod_"
            "*/*spearmint_gitfork_mod"
            ".pkl")
            ])

bootstraps =  (2,) # 5) #, 10)
distance = ("l1", )#"l2")
metafeature_subset = mf_module.subsets
metafeature_subset = dict()
metafeature_subset["all"] = set(mf_module.metafeatures.functions.keys())
metafeature_subset["pfahringer_2000_experiment1"] = set(["number_of_features",
                                             "number_of_numeric_features",
                                             "number_of_categorical_features",
                                             "number_of_classes",
                                             "class_probability_max",
                                             "landmark_lda",
                                             "landmark_naive_bayes",
                                             "landmark_decision_tree"])

for num_bootstrap, dist, subset in itertools.product(
                bootstraps, distance, metafeature_subset, repeat=1):
    optimizers["Spearmint/Meta%d_%s_%s" % (num_bootstrap, dist, subset)] = \
        "%s" + "/bootstrapped%d_%s_%sspearmint_gitfork_mod_" \
        "*/*spearmint_gitfork_mod.pkl" % (num_bootstrap, dist, subset)

print optimizers

rankings = np.zeros((50, len(optimizers)), dtype=np.float64)
summed_wins_of_optimizer = [np.zeros((len(optimizers.keys()), len(optimizers.keys())))
                            for i in range(51)]
summed_wins_of_optimizer_rounded = [np.zeros((len(optimizers.keys()), len(optimizers.keys())))
                                    for i in range(51)]


num_datasets = 0

num_folds = 1 # 3
# Plot average error over all datasets
gigantic_pickle_list = [[] for optimizer in optimizers]
for idx, dataset in enumerate(datasets):
    dataset_rankings = np.zeros((50, len(optimizers)), dtype=np.float64)
    for fold in range(num_folds):
        dataset_dir = "did_%d_fold%d" % (dataset, fold)
        exp_dir = os.path.join(experiments_directory, dataset_dir)
        argument_list = []
        for optimizer in optimizers:
            pkls = glob.glob(optimizers[optimizer] % exp_dir)
            argument_list.append(optimizer)
            argument_list.extend(pkls)
        pkl_list_main, name_list_main = plot_util.get_pkl_and_name_list(argument_list)
        for i, optimizer in enumerate(optimizers):
            gigantic_pickle_list[i].extend(pkl_list_main[i])

for idx, dataset in enumerate(datasets):
    dataset_rankings = np.zeros((50, len(optimizers)), dtype=np.float64)
    for fold in range(num_folds):
        dataset_dir = "%s/did_%d_fold%d" % (experiments_directory, dataset, fold)
        plot_suffix = "did_%d_fold%d" % (dataset, fold)
        if not os.path.isdir(dataset_dir) or "did" not in dataset_dir:
            continue

        num_datasets += 1
        print "%d/%d %s" % (idx, len(datasets) * num_folds, dataset_dir)
        argument_list = []
        for optimizer in optimizers:
            pkls = glob.glob(optimizers[optimizer] % dataset_dir)
            argument_list.append(optimizer)
            argument_list.extend(pkls)

        pkl_list_main, name_list_main = plot_util.get_pkl_and_name_list(argument_list)

        ground_truth = "%s/did_%d_fold%d/gridsearch_*/gridsearch.pkl" % \
                             (ground_truth_dir, dataset, fold)
        grid_pkl = glob.glob(ground_truth)

        ########################################################################
        # Create useless and gigantic latex table
        experiment_name = os.path.split(dataset_dir)[1].replace("_", "\_")

        header = generate_tex_table.main(pkl_list_main + [grid_pkl],
                                         name_list_main + [['Gridsearch', 1]],
                                         cut=sys.maxint,
                                         template_string=header_template,
                                         experiment_name=experiment_name)
        fh.write(header)
        fh.write("\n")

        for i in [1, 2, 5, 10, 20, 50]:
            fh.write(generate_tex_table.main(pkl_list_main, name_list_main,
                                                cut=i,
                                                template_string=line_template,
                                                experiment_name=experiment_name,
                                                num_evals=str(i)))
            fh.write("\n")

        fh.write(generate_tex_table.main(pkl_list_main + [grid_pkl],
                                         name_list_main + [['Gridsearch', 1]],
                                         cut=sys.maxint,
                                         template_string=line_template,
                                         experiment_name=experiment_name,
                                         num_evals="MAX"))
        fh.write("\n")
        fh.write('\\bottomrule\n\end{tabularx}\n\\caption{ %s }\n\end{'
                 'table}\n' % experiment_name)

        if (idx + 1) % 12 == 0:
            fh.write("\clearpage\n")
        fh.flush()

        ########################################################################
        # Plot error traces for one dataset
        plotTraceWithStd_perEval.main(pkl_list_main, name_list_main, True,
                                 save="%s/error_trace_%s.png" % (plot_dir, plot_suffix))

        print grid_pkl
        with open(grid_pkl[0]) as grid_fh:
            grid_pkl_open = cPickle.load(grid_fh)
        optimum = plot_util.get_best(grid_pkl_open)
        plotTraceWithStd_perEval.main(pkl_list_main, name_list_main, True,
                                      optimum=optimum, log=True,
                                 save="%s/optimum_error_trace_%s.png" % (
                                     plot_dir, plot_suffix))

        ########################################################################
        # Statistical stuff for one dataset
        for cut in range(1, 51):
            tmp, wins_of_optimizer, wins_of_optimizer_rounded = \
                statistics.main(pkl_list_main, name_list_main, cut=cut)
            summed_wins_of_optimizer[cut] += wins_of_optimizer
            summed_wins_of_optimizer_rounded[cut] += wins_of_optimizer_rounded


        ########################################################################
        # draw a ranking graph over one datasets
        ranking = np.zeros((50, len(optimizers)), dtype=np.float64)
        ranking_2 = np.zeros((50, len(optimizers)), dtype=np.float64)
        pickles = plot_util.load_pickles(name_list_main, pkl_list_main)
        for i in range(ranking.shape[0]):
            best_dict, idx_dict, keys = plot_util.get_best_dict(name_list_main,
                                                                pickles, i)
            best_list_averaged = [np.mean(best_dict[optimizer]) for optimizer in optimizers]
            ranks_over_averages = scipy.stats.rankdata(best_list_averaged)
            for j, optimizer in enumerate(optimizers):
                ranking[i][j] = ranks_over_averages[j]
                rankings[i][j] += ranks_over_averages[j]
                dataset_rankings[i][j] += ranks_over_averages[j]

            best_list = []
            num_runs_per_optimizer = dict()
            for optimizer in optimizers:
                num_runs_per_optimizer[optimizer] = len(best_dict[optimizer])
                best_list.extend(best_dict[optimizer])
            ranks = scipy.stats.rankdata(best_list)
            average_over_ranks = []
            offset = 0
            for j, optimizer in enumerate(optimizers):
                m = np.mean(ranks[offset:offset+num_runs_per_optimizer[optimizer]])
                ranking_2[i][j] = m
                offset += num_runs_per_optimizer[optimizer]

        fig = plt.figure()
        ax = plt.subplot(111)
        colors = plot_util.get_plot_colors()
        for i, optimizer in enumerate(optimizers):
            ax.plot(range(1, 51), ranking[:, i], color=colors.next(), alpha=0.9,
                    label=optimizer.replace("\\", ""))

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=4, labelspacing=0.25, fontsize=12)
        box = ax.get_position()

        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
        plt.savefig("%s/ranks_over_averages_%s.png" % (plot_dir, plot_suffix))
        plt.close(fig)

        fig = plt.figure()
        ax = plt.subplot(111)
        colors = plot_util.get_plot_colors()
        for i, optimizer in enumerate(optimizers):
            ax.plot(range(1, 51), ranking_2[:, i], color=colors.next(), alpha=0.9,
                    label=optimizer.replace("\\", ""))

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=4, labelspacing=0.25, fontsize=12)
        box = ax.get_position()

        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
        plt.savefig("%s/average_over_ranks_%s.png" % (plot_dir, plot_suffix))
        plt.close(fig)

    ############################################################################
    # Plot a ranking over the average of all folds of a dataset
    dataset_rankings = dataset_rankings / float(num_folds)
    fig = plt.figure()
    ax = plt.subplot(111)
    colors = plot_util.get_plot_colors()
    for i, optimizer in enumerate(optimizers):
        ax.plot(range(1, 51), dataset_rankings[:, i], color=colors.next(), alpha=0.9,
                label=optimizer.replace("\\", ""))

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=4, labelspacing=0.25, fontsize=12)
    box = ax.get_position()

    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
    plt.savefig("%s/dataset_average_%s.png" % (plot_dir, dataset))
    plt.close(fig)

################################################################################
# print all statistics
print "------------------------------------------------------------------------"
print optimizers.keys()
with open("%s/sums_of_wins.tex" % plot_dir, "w") as fh2:
    tmp = np.round(summed_wins_of_optimizer[-1] / len(datasets) * 100, 1)
    fh2.write(tex_header.replace("a2", "a4").replace("a4paper", "landscape,a4paper"))
    fh2.write("\\begin{table}[t]\n\\begin{tabularx}{\\textwidth}{l|" +
              "X" * len(optimizers.keys()) + "|}")
    for key in optimizers.keys():
        fh2.write(" & ")
        fh2.write("{\\rotatebox{90}{%s}}" % key.replace("_", "\_"))
    fh2.write(" \\\\\n")
    fh2.write("\\toprule\n")

    for idx, line in enumerate(tmp):
        fh2.write(optimizers.keys()[idx].replace("_", "\_"))
        for column in line:
            fh2.write(" & ")
            fh2.write("%3.2f" % column)
        fh2.write(" \\\\\n")

    fh2.write('\\bottomrule\n\end{tabularx}\n\\caption{ %s }\n\end{'
             'table}\n' % "Wins of row against column.")
    fh2.write("\end{document}\n")

with open("%s/sums_of_wins_rounded.tex" % plot_dir, "w") as fh2:
    tmp = np.round(summed_wins_of_optimizer_rounded[-1] / len(datasets) * 100, 1)
    fh2.write(tex_header.replace("a2", "a4").replace("a4paper", "landscape,a4paper"))
    fh2.write("\\begin{table}[t]\n\\begin{tabularx}{\\textwidth}{l|" +
              "X" * len(optimizers.keys()) + "|}")
    for key in optimizers.keys():
        fh2.write(" & ")
        fh2.write("{\\rotatebox{90}{%s}}" % key.replace("_", "\_"))
    fh2.write(" \\\\\n")
    fh2.write("\\toprule\n")

    for idx, line in enumerate(tmp):
        fh2.write(optimizers.keys()[idx].replace("_", "\_"))
        for column in line:
            fh2.write(" & ")
            fh2.write("%3.2f" % column)
        fh2.write(" \\\\\n")

    fh2.write('\\bottomrule\n\end{tabularx}\n\\caption{ %s }\n\end{'
             'table}\n' % "Wins of row against column.")
    fh2.write("\end{document}\n")

################################################################################
# Plot statistics
for opt1_idx, key in enumerate(optimizers):
    plt.figure(dpi=600, figsize=(16, 9))
    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, dpi=600,
                                        figsize=(16, 9))
    colors = plot_util.get_plot_colors()

    for opt2_idx, key2 in enumerate(optimizers):
        if opt1_idx == opt2_idx:
            continue

        y = []
        y1 = []
        for i in range(0, 51):
            y.append(summed_wins_of_optimizer[i][opt1_idx, opt2_idx]
                     / len(datasets))
            y1.append(- summed_wins_of_optimizer[i][opt2_idx, opt1_idx]
                      / len(datasets))

        label = "%s vs %s" % (key, key2)
        color = colors.next()
        ax0.plot(range(0, 51), y, color=color, alpha=0.9, label=label,
                 linewidth=2)
        ax1.plot(range(0, 51), y1, color=color, alpha=0.9, label=label,
                 linewidth=2)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", fancybox=True, ncol=2,
               shadow=True)

    ax0.set_xlim((0, 50))
    ax0.set_ylim((0, 1))
    ax1.set_xlim((0, 50))
    ax1.set_ylim((-1, 0))

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig("%s/percentage_of_wins_%s.png" %
                (plot_dir, key.replace("/", "--")))
    plt.close(fig)

################################################################################
# draw a ranking graph averaged over all datasets
plt.figure()
ax = plt.subplot(111)
colors = plot_util.get_plot_colors()
rankings = rankings / float(num_datasets)
for i, optimizer in enumerate(optimizers):
        ax.plot(range(1, 51), rankings[:, i], color=colors.next(),
        label=optimizer.replace("\\", ""))

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=4, labelspacing=0.25, fontsize=12)
box = ax.get_position()

ax.set_position([box.x0, box.y0 + box.height * 0.1,
             box.width, box.height * 0.9])
plt.savefig("%s/all_datasets.png" % plot_dir)


plotTraceWithStd_perEval.main(gigantic_pickle_list, name_list_main, True,
                              save="%s/all_datasets_error.png" % plot_dir)
plotTraceWithStd_perEval.main(gigantic_pickle_list, name_list_main, True,
                              save="%s/all_datasets_log_error.png" % plot_dir,
                              log=True)


fh.write("\end{document}\n")
fh.close()