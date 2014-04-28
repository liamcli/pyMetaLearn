from argparse import ArgumentParser
from collections import OrderedDict
import glob
import numpy as np
import os
import re
import scipy.stats
import sys

from matplotlib import pyplot as plt

import HPOlib.Plotting.plot_util as plot_util
import HPOlib.Plotting.generateTexTable as generate_tex_table
import HPOlib.Plotting.plotTraceWithStd_perEval as plotTraceWithStd_perEval

header_template = """\\begin{table}[t]
\\begin{tabularx}{\\textwidth}{lr{%- for name in result_values -%}|Xr{%- endfor -%}}
\\toprule
\multicolumn{2}{l}{}
{%- for name in result_values -%}
&\multicolumn{2}{c}{\\bf {{name}}}
{%- endfor -%}
\\\\
\\multicolumn{1}{l}{\\bf Experiment} &\multicolumn{1}{r}{\\#evals}
{%- for name in result_values -%}
&\\multicolumn{1}{l}{Valid.\\ loss} &\\multicolumn{1}{r}{Best loss}
{%- endfor -%}
\\\\
\\toprule
"""

line_template = """{{ experiment }} & {{ evals }}
{%- for name in result_values -%}
{%- set results = result_values[name] -%}
{{ ' & ' }}{% if results['mean_best'] == True %}\\textbf{ {%- endif %}{{results['mean']|round(3, 'floor') }}{% if results['mean_best'] == True %}}{% endif %}$\\pm${{ results['std']|round(3, 'floor')}} & {{results['min']|round(3, 'floor') }}{%- endfor %} \\\\
"""

fh = open("giant_tex_table_py.tex", "w")
fh.write("""\documentclass{article} % For LaTeX2

\usepackage[a2paper, left=5mm, right=5mm, top=5mm, bottom=5mm]{geometry}
\usepackage{multirow}           % import command \multicolmun
\usepackage{tabularx}           % Convenient table formatting
\usepackage{booktabs}           % provides 	oprule, \midrule andottomrule

\\begin{document}

""")

parser = ArgumentParser()
args, unknown = parser.parse_known_args()

try:
    os.mkdir("plots")
except:
    pass

datasets = os.listdir(".")
dataset_stems = set()
for dataset in datasets:
    match = re.match(r"([A-Za-z0-9]*_[A-Za-z0-9]*)_(fold[0-9])", dataset)
    if match:
        dataset_stems.add(match.group(1))
datasets = list(dataset_stems)
print datasets

num_folds = 3

datasets.sort()
optimizers = OrderedDict([("Meta\_VSM",
                  "%s/metalearn_optimizer_10000_*/metalearn_optimizer.pkl"),
                         ("Meta\_L1",
                  "%s/metalearn_optimizer_10001_*/metalearn_optimizer.pkl"),
                         ("Meta\_L2",
                  "%s/metalearn_optimizer_10002_*/metalearn_optimizer.pkl"),
                         ("SMAC", "%s/smac_2_06_01-dev_*/smac_2_06_01-dev"
                                  ".pkl"),
           #              ("Spearmint",
           #       "%s/spearmint_april2013_mod_*/spearmint_april2013_mod.pkl"),
                         ("Spearmint/Grid",
            "%s/spearmint_gitfork_mod_vanilla_*/spearmint_gitfork_mod.pkl"),
                         ("Spearmint/Meta2",
            "%s/spearmint_gitfork_mod_bootstrap2_*/spearmint_gitfork_mod.pkl"),
                         ("Spearmint/Meta5",
            "%s/spearmint_gitfork_mod_bootstrap5_*/spearmint_gitfork_mod.pkl"),
                         ("Spearmint/Meta10",
            "%s/spearmint_gitfork_mod_bootstrap10_*/spearmint_gitfork_mod.pkl"),
                         ("random", "%s/random_hyperopt_august2013_mod"
                        "*/random_hyperopt_august2013_mod.pkl"),
                         ("TPE",
    "%s/hyperopt_august2013_mod_*/hyperopt_august2013_mod" \
                     ".pkl")])
rankings = np.zeros((50, len(optimizers)), dtype=np.float64)
num_datasets = 0

# Plot average error over all datasets
gigantic_pickle_list = [[] for optimizer in optimizers]
for idx, dataset in enumerate(datasets):
    dataset_rankings = np.zeros((50, len(optimizers)), dtype=np.float64)
    for fold in range(num_folds):
        dataset_dir = "%s_fold%d" % (dataset, fold)
        argument_list = []
        for optimizer in optimizers:
            pkls = glob.glob(optimizers[optimizer] % dataset_dir)
            argument_list.append(optimizer)
            argument_list.extend(pkls)
        pkl_list_main, name_list_main = plot_util.get_pkl_and_name_list(argument_list)
        for i, optimizer in enumerate(optimizers):
            gigantic_pickle_list[i].extend(pkl_list_main[i])
plotTraceWithStd_perEval.main(gigantic_pickle_list, name_list_main, True,
                              save="plots/all_datasets_error.png")
plotTraceWithStd_perEval.main(gigantic_pickle_list, name_list_main, True,
                              save="plots/all_datasets_log_error.png",
                              log=True)


for idx, dataset in enumerate(datasets):
    dataset_rankings = np.zeros((50, len(optimizers)), dtype=np.float64)
    for fold in range(num_folds):
        dataset_dir = "%s_fold%d" % (dataset, fold)
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

        grid_pkl = glob.glob("/home/feurerm/thesis/experiments" \
                     "/2014_02_20_gather_metadata_from_openml/%s/gridsearch_" \
                     "*/gridsearch.pkl" % dataset_dir.replace("_fold", "fold"))

        ########################################################################
        # Create useless and gigantic latex table
        header = generate_tex_table.main(pkl_list_main + [grid_pkl],
                                         name_list_main + [['Gridsearch', 1]],
                                         cut=sys.maxint,
                                         template_string=header_template)
        fh.write(header)
        fh.write("\n")

        experiment_name = dataset_dir.replace("_", "\_")

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
        fh.write('\\bottomrule\n\end{tabularx}\n\end{table}\n')

        if (idx + 1) % 12 == 0:
            fh.write("\clearpage\n")
        fh.flush()

        ########################################################################
        # Plot error traces for one dataset
        plotTraceWithStd_perEval.main(pkl_list_main, name_list_main, True,
                                 save="plots/error_trace_%s.png" % dataset_dir)

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
        markers = plot_util.get_plot_markers()
        for i, optimizer in enumerate(optimizers):
            ax.plot(range(1, 51), ranking[:, i], color=colors.next(), alpha=0.9,
                    marker=markers.next(), label=optimizer.replace("\\", ""))

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=4, labelspacing=0.25, fontsize=12)
        box = ax.get_position()

        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
        plt.savefig("plots/ranks_over_averages_%s.png" % dataset_dir)
        plt.close(fig)

        fig = plt.figure()
        ax = plt.subplot(111)
        colors = plot_util.get_plot_colors()
        markers = plot_util.get_plot_markers()
        for i, optimizer in enumerate(optimizers):
            ax.plot(range(1, 51), ranking_2[:, i], color=colors.next(), alpha=0.9,
                    marker=markers.next(), label=optimizer.replace("\\", ""))

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=4, labelspacing=0.25, fontsize=12)
        box = ax.get_position()

        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
        plt.savefig("plots/average_over_ranks_%s.png" % dataset_dir)
        plt.close(fig)

    ############################################################################
    # Plot a ranking over the average of all folds of a dataset
    dataset_rankings = dataset_rankings / float(num_folds)
    fig = plt.figure()
    ax = plt.subplot(111)
    colors = plot_util.get_plot_colors()
    markers = plot_util.get_plot_markers()
    for i, optimizer in enumerate(optimizers):
        ax.plot(range(1, 51), dataset_rankings[:, i], color=colors.next(), alpha=0.9,
                marker=markers.next(), label=optimizer.replace("\\", ""))

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=4, labelspacing=0.25, fontsize=12)
    box = ax.get_position()

    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
    plt.savefig("plots/dataset_average_%s.png" % dataset)
    plt.close(fig)


################################################################################
# draw a ranking graph averaged over all datasets
plt.figure()
ax = plt.subplot(111)
colors = plot_util.get_plot_colors()
markers = plot_util.get_plot_markers()
rankings = rankings / float(num_datasets)
for i, optimizer in enumerate(optimizers):
        ax.plot(range(1, 51), rankings[:, i], color=colors.next(),
        marker=markers.next(), label=optimizer.replace("\\", ""))

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=4, labelspacing=0.25, fontsize=12)
box = ax.get_position()

ax.set_position([box.x0, box.y0 + box.height * 0.1,
             box.width, box.height * 0.9])
plt.savefig("plots/all_datasets.png")

fh.write("\end{document}\n")
fh.close()