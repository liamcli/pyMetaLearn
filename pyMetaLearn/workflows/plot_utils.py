import cPickle
import glob
import itertools
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import sys

import HPOlib.Plotting.plot_util as plot_util
import HPOlib.Plotting.statistics as statistics

def find_ground_truth(globstring):
    glob_results = glob.glob(globstring)
    if len(glob_results) > 1:
        raise Exception("There must be only one ground truth directory for "
                        "%s" % globstring)
    elif len(glob_results) == 0:
        print "Found no ground truth for %s" % globstring
        return None

    with open(glob_results[0]) as fh:
        trials = cPickle.load(fh)

    return trials


def plot_rankings(trial_list, name_list, optimum=0, title="", log=False,
                  save="", y_min=0, y_max=0, cut=sys.maxint, figsize=(16, 6),
                  legend_ncols=4):
    # check if all optimizers have the same number of runs
    if np.mean([name[1] for name in name_list]) != name_list[0][1]:
        raise Exception("All optimizers must have the same numbers of "
                        "experiment runs! %s" % name_list)
    num_runs = name_list[0][1]

    optimizers = [name[0] for name in name_list]
    pickles = plot_util.load_pickles(name_list, trial_list)
    length = len(plot_util.extract_trajectory(pickles[optimizers[0]][0]))
    ranking = np.zeros((length, len(name_list)), dtype=np.float64)

    num_products = 0
    for product in itertools.product(range(num_runs), repeat=len(optimizers)):
        num_products += 1

    keep_probability = 1.0
    if num_products > 1000:
        keep_probability = 1000. / float(num_products)

    randomness = np.random.random(1000000)
    print keep_probability

    for i in range(ranking.shape[0]):
        num_products = 0

        # TODO: this should be much faster with Cython
        j = 0   # should be faster than indexing with modulos
        for product in itertools.product(range(num_runs), repeat=len(optimizers)):
            j += 1

            if j >= 999999:
                j = 0

            if randomness[j] > keep_probability:
                continue

            ranks = scipy.stats.rankdata(
                [np.round(plot_util.get_best(pickles[optimizers[idx]][number], i), 5)
                 for idx, number in enumerate(product)])
            num_products += 1
            for j, optimizer in enumerate(optimizers):
                ranking[i][j] += ranks[j]

        for j, optimizer in enumerate(optimizers):
            ranking[i][j] = ranking[i][j] / num_products

    fig = plt.figure(dpi=600, figsize=figsize)
    ax = plt.subplot(111)
    colors = plot_util.get_plot_colors()
    for i, optimizer in enumerate(optimizers):
        ax.plot(range(1, length+1), ranking[:, i], color=colors.next(),
                linewidth=3, label=optimizer.replace("\\", ""))

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=legend_ncols,
              labelspacing=0.25, fontsize=12)
    ax.set_xlabel("\#Function evaluations")
    ax.set_ylabel("Average rank")
    box = ax.get_position()

    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    if save != "":
        plt.savefig(save, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, bbox_inches="tight", pad_inches=0.1)
    else:
        plt.show()
    plt.close(fig)
    return ranking


def plot_summed_wins_of_optimizers(trial_list_per_dataset, name_list,
                                   save="",  cut=sys.maxint, dpi=600,
                                   figsize=(16, 6), legend_ncols=3):

    with open(trial_list_per_dataset[0][0][0]) as fh:
        probing_trial = cPickle.load(fh)
    cut = min(cut, len(probing_trial['trials']))
    optimizers = [name[0] for name in name_list]

    if cut == sys.maxint:
        raise ValueError("You must specify a cut value beforehand!")

    summed_wins_of_optimizer = \
        [np.zeros((len(optimizers), len(optimizers))) for i in range(cut+1)]

    for i, pkl_list in enumerate(trial_list_per_dataset):
        ########################################################################
        # Statistical stuff for one dataset
        for c in range(1, cut+1):
            wins_of_optimizer = statistics.get_pairwise_wins(pkl_list, name_list, cut=c)

            for opt1_idx, key in enumerate(optimizers):
                for opt2_idx, key2 in enumerate(optimizers):
                    summed_wins_of_optimizer[c][opt1_idx][opt2_idx] += \
                        wins_of_optimizer[key][key2]

    ################################################################################
    # Plot statistics
    for opt1_idx, key in enumerate(optimizers):
        fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, dpi=dpi,
                                       figsize=figsize)
        colors = plot_util.get_plot_colors()
        y_max = 0.

        for opt2_idx, key2 in enumerate(optimizers):
            if opt1_idx == opt2_idx:
                continue

            y = []
            y1 = []
            for i in range(0, cut+1):
                y.append(summed_wins_of_optimizer[i][opt1_idx, opt2_idx]
                         / len(trial_list_per_dataset))
                y1.append(- summed_wins_of_optimizer[i][opt2_idx, opt1_idx]
                          / len(trial_list_per_dataset))

            y_max_tmp = max(np.max(y), np.max(np.abs(y1)))
            y_max_tmp = np.ceil(y_max_tmp * 10) / 10.
            y_max = max(y_max_tmp, y_max)

            label = "%s vs %s" % (key, key2)
            color = colors.next()
            ax0.plot(range(0, cut+1), y, color=color, label=label, linewidth=3)
            ax1.plot(range(0, cut+1), y1, color=color, label=label, linewidth=3)

        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", fancybox=True,
                   ncol=legend_ncols, shadow=True)

        ax0.set_xlim((0, cut))
        ax0.set_ylim((0, y_max))
        ax1.set_xlim((0, cut))
        ax1.set_ylim((-y_max, 0))

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.xlabel("\#Function evaluations")
        if save != "":
            plt.savefig(save % key, facecolor='w', edgecolor='w',
                        orientation='portrait', papertype=None, format=None,
                        transparent=False, bbox_inches="tight", pad_inches=0.1)
        else:
            plt.show()
        plt.close(fig)