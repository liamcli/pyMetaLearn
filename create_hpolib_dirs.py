import argparse
import cPickle
import os
import shutil

import numpy as np

import sklearn.cross_validation


config_template = \
"""[DEFAULT]
function = %s
numberOfJobs = %d
result_on_terminate = %f
numberCV = %d
    """
def configure_config_template(function, number_of_jobs=200,
                              result_on_terminate=1,
                              number_cv=10):
    return config_template % (function, number_of_jobs, result_on_terminate,
                              number_cv)


def create_hpolib_dir(dataset, file_dir, template_dir, target_dir):
    dataset_dir = os.path.abspath(os.path.join(file_dir, dataset))

    if target_dir is None:
        target_dir = os.path.join(dataset_dir, "experiments")
    elif not os.path.abspath(target_dir):
        target_dir = os.path.join(dataset_dir, target_dir)
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    fh = open(os.path.join(dataset_dir, dataset + ".pkl"))
    ds = cPickle.load(fh)
    fh.close()
    X, Y = ds.get_processed_files()
    kf = sklearn.cross_validation.StratifiedKFold(Y, n_folds=3, indices=True)
    for i, splits  in enumerate(kf):
        train_split, test_split = splits
        output_dir = os.path.join(target_dir, os.path.basename(dataset) +
                                  "fold%d" % i)
        shutil.copytree(template_dir, output_dir, symlinks=True)

        np.save(os.path.join(output_dir, "train.npy"), X[train_split])
        np.save(os.path.join(output_dir, "train_targets.npy"), Y[train_split])
        np.save(os.path.join(output_dir, "test.npy"), X[test_split])
        np.save(os.path.join(output_dir, "test_targets.npy"), Y[test_split])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs=1, help="Directory of dataset "
                                                   "directories files.")
    parser.add_argument("template_dir", nargs=1)
    parser.add_argument("-t", "--target_dir")
    args = parser.parse_args()

    dataset_directory = args.directory[0]
    template_dir = args.template_dir[0]

    directory_content = os.listdir(dataset_directory)
    directory_content.sort()
    for dataset in directory_content:
        create_hpolib_dir(dataset, dataset_directory, template_dir,
              args.target_dir)