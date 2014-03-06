import argparse
import cPickle
import itertools
import os

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import metafeatures


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


def plot_metafeatures(dataset_directory):
    directory_content = os.listdir(dataset_directory)
    directory_content.sort()
    if os.path.exists("../figures/meta_feature_array.npy"):
        meta_feature_array = np.load("../figures/meta_feature_array.npy")
    else:
        meta_feature_array = np.ndarray((len(directory_content),
                                         len(metafeatures.metafeatures.functions)),
                                         dtype=np.float64)
        for i, dataset in enumerate(directory_content):
            print dataset
            X, Y = load_dataset(dataset, dataset_directory)
            mf = metafeatures.calculate_all_metafeatures(X, Y)
            for j, key in enumerate(mf):
                meta_feature_array[i][j] = mf[key]
        np.save("../figures/meta_feature_array.npy", meta_feature_array)
    X_min = np.nanmin(meta_feature_array, axis=0)
    X_max = np.nanmax(meta_feature_array, axis=0)
    normalized_meta_feature_array = (meta_feature_array - X_min) / (X_max -
                                                                    X_min)
    pca = PCA(2)
    transformation = pca.fit_transform(normalized_meta_feature_array)
    #for i, dataset in enumerate(directory_content):
    #    print dataset, meta_feature_array[i]
    labels = directory_content
    fig = plt.figure(dpi=300, figsize=(16, 9))
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
    plt.savefig("../figures/pca.png")
    plt.savefig("../figures/pca.pdf")
    keys = dict([(key, i) for i, key in enumerate(metafeatures.metafeatures
                                                  .functions.keys())])
    for mf_1, mf_2 in itertools.combinations(metafeatures.metafeatures
                                                     .functions.keys(), 2):
        x = meta_feature_array[:, keys[mf_1]]
        y = meta_feature_array[:, keys[mf_2]]
        print len(x), x
        print len(y), y
        plt.figure()
        plt.plot(x, y, "x")
        plt.xlabel(mf_1)
        plt.ylabel(mf_2)
        plt.savefig("../figures/" + mf_1 + "__" + mf_2 + ".png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs=1, help="Directory of dataset "
                                                   "directories files.")
    args = parser.parse_args()

    dataset_directory = args.directory[0]

    plot_metafeatures()


