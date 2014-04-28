import cPickle
import csv
import itertools
import os

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scipy.stats

import pyMetaLearn.metafeatures.metafeatures as metafeatures
import pyMetaLearn.openml.manage_openml_data


def load_dataset(dataset, dataset_directory):
    dataset_dir = os.path.abspath(os.path.join(dataset_directory, dataset))
    fh = open(os.path.join(dataset_dir, dataset + ".pkl"))
    ds = cPickle.load(fh)
    fh.close()
    data_frame = ds.convert_arff_structure_to_pandas(ds
                                                     .get_unprocessed_files())
    class_ = data_frame.keys()[-1]
    attributes = data_frame.keys()[0:-1]
    X = data_frame[attributes]
    Y = data_frame[class_]
    return X, Y


def plot_metafeatures(dataset_directory, target_directory):
    datasets = pyMetaLearn.openml.manage_openml_data.get_local_datasets()
    to_use = list()
    for r in [range(2, 17), (18,), range(20, 25), range(26, 27), range(28, 38), \
        range(39, 46), range(47, 52), range(53, 57), range(58, 63)]:
        to_use.extend(r)
    to_use = set(to_use)

    labels = []
    metafeature_names = []
    csv_file = open(os.path.join(target_directory, "metafeatures.csv"), "w")
    csv_writer = csv.writer(csv_file)
    csv_header_written = False

    meta_feature_array = np.ndarray((len(to_use),
                                     len(metafeatures.metafeatures.functions)),
                                     dtype=np.float64)

    idx = 0
    for did in datasets:
        if did not in to_use:
            continue

        dataset = pyMetaLearn.openml.manage_openml_data\
            .get_local_dataset(did)
        print did, dataset._name
        labels.append(dataset._name)

        X, Y = dataset.get_npy(scaling="scale")
        print "X", np.isfinite(X).all()
        print "Y", np.isfinite(Y).all()

        mf = dataset.get_metafeatures()
        for j, item in enumerate(mf.items()):
            key, value = item
            meta_feature_array[idx][j] = value
            if not csv_header_written:
                metafeature_names.append(key)

        if not csv_header_written:
            csv_writer.writerow(metafeature_names)
            csv_header_written = True
        csv_writer.writerow(meta_feature_array[idx])
        csv_file.flush()

        idx += 1

    csv_file.close()

    X_min = np.nanmin(meta_feature_array, axis=0)
    X_max = np.nanmax(meta_feature_array, axis=0)
    normalized_meta_feature_array = (meta_feature_array - X_min) / (X_max -
                                                                    X_min)

    # PCA
    pca = PCA(2)
    transformation = pca.fit_transform(normalized_meta_feature_array)
    #for i, dataset in enumerate(directory_content):
    #    print dataset, meta_feature_array[i]
    fig = plt.figure(dpi=600, figsize=(16, 9))
    ax = plt.subplot(111)
    for i, data in enumerate(
            zip(labels, transformation[:, 0], transformation[:, 1])):
        label, x, y = data
        ax.scatter(x, y, marker='$%i$' % i, label=label, s=30, linewidths=0.1)
        #plt.annotate(label, xy=(x, y), size=10)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
    ax.legend(loc=2, borderaxespad=0., bbox_to_anchor=(1.05, 1),
              fancybox=True, shadow=True, fontsize=10, ncol=2)
    plt.savefig(os.path.join(target_directory, "pca.png"))
    plt.savefig(os.path.join(target_directory, "pca.pdf"))
    plt.clf()

    # Relation of features to each other...
    correlations = []
    keys = dict([(key, idx) for idx, key in enumerate(metafeature_names)])
    for mf_1, mf_2 in itertools.combinations(metafeature_names, 2):

        x = meta_feature_array[:, keys[mf_1]]
        y = meta_feature_array[:, keys[mf_2]]
        rho, p = scipy.stats.spearmanr(x, y)
        correlations.append((rho, "%s-%s" % (mf_1, mf_2)))

        plt.figure()
        plt.plot(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01))
        plt.plot(x, y, "x")
        plt.xlabel(mf_1)
        plt.ylabel(mf_2)
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        plt.savefig(os.path.join(target_directory, mf_1 + "__" + mf_2 + ".png"))

    correlations.sort()
    for cor in correlations:
        print cor


if __name__ == "__main__":
    # TODO: make this more generic...
    experiments_directory = \
        "/home/feurerm/thesis/experiments/2014_04_17_gather_new_metadata_from_openml"
    pyMetaLearn.openml.manage_openml_data.set_local_directory(
        "/home/feurerm/thesis/datasets/openml/")
    local_directory = pyMetaLearn.openml.manage_openml_data.get_local_directory()
    dataset_dir = os.path.join(local_directory, "datasets")
    target_dir = os.path.join(local_directory, "metafeature_plots")

    try:
        os.mkdir(target_dir)
    except:
        pass

    plot_metafeatures(dataset_dir, target_dir)


