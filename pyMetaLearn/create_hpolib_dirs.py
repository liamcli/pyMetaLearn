import argparse
import cPickle
import os
import shutil

import numpy as np

import sklearn.cross_validation


config_template = \
"""[HPOLIB]
function = %s
number_of_jobs = %d
result_on_terminate = %f
number_cv_folds = %d
    """
def configure_config_template(function, number_of_jobs=200,
                              result_on_terminate=1,
                              number_cv=10):
    return config_template % (function, number_of_jobs, result_on_terminate,
                              number_cv)


def create_hpolib_dir(dataset, file_dir, template_dir, target_dir):
    """Create an experiment directory for the HPOlib package. Copies stuff
    from the template directories to the target directories.

    Inputs:
    * dataset: the name of the dataset
    * file_dir: the directory the dataset resides in
    * template_dir: directory of the HPOlib experiment template
    * target_dir: target for the created experiment directory

    """
    dataset_dir = os.path.abspath(os.path.join(file_dir, dataset))

    if target_dir is None:
        target_dir = os.path.join(dataset_dir, "experiments")
    elif not os.path.abspath(target_dir):
        target_dir = os.path.join(dataset_dir, target_dir)
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    for i in range(3):
        output_dir = os.path.join(target_dir, os.path.basename(dataset) +
                                  "_fold%d" % i)
        shutil.copytree(template_dir, output_dir, symlinks=True)


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