"""
This file is a wrapper for all classification tasks from the skdata repository by James Bergstra:
https://github.com/jaberg/skdata

Exceptions are
- Labelled Faces in the Wild: No classification task
- PosnerKeele1963E3: Data is generated
- Austin Open Data: Data is downloaded from the internet
- Van Hateren Image Dataset: Don't actually know what to do with this dataset
- iicbu datasets: Too large
- Pascal: Object recognition is a different task
- Pubfig: This is a face recognition task
- KaggleFinalCompetition: This is a dataset for which the user has to be logged in
- Iris: Already present in OpenML
- Diabetes: Already in OpenML
- Digits: Already in OpenML
- Brodatz: No view available
- Caltech 101: No view available
- Caltech 256: No view available
"""
import arff
from collections import OrderedDict
import cPickle
import gzip
import itertools
import os
import time
import types

import numpy as np
import sklearn.metrics
import sklearn.utils
import sklearn.svm
import sklearn.preprocessing
import sklearn.ensemble
import skdata
import skdata.base
from skdata.base import Task


# TODO: add PIL to dependencies
print skdata
import skdata.cifar10
import skdata.iris
import skdata.larochelle_etal_2007
print skdata.larochelle_etal_2007
import skdata.larochelle_etal_2007.dataset
import skdata.larochelle_etal_2007.view
import skdata.mnist
import skdata.svhn
import skdata.brodatz
import skdata.caltech
import skdata.diabetes
import skdata.digits
import skdata.pubfig83

import pyMetaLearn.openml.manage_openml_data
from pyMetaLearn.openml.openml_dataset import OpenMLDataset
from pyMetaLearn.openml.openml_task import OpenMLTask


def prepare(self):
    """
    This is modification from skdata/larochelle_et_al_2007/view.py and will be
    injected in there instead of the original protocol
    """

    ds = self.dataset
    meta = ds.build_meta()

    n_train = ds.descr['n_train']
    n_valid = ds.descr['n_valid']
    n_test = ds.descr['n_test']

    start = 0
    end = n_train
    self.train = Task('vector_classification',
                      name='train',
                      x=ds._inputs[start:end].reshape(end-start, -1),
                      y=ds._labels[start:end],
                      n_classes=ds.descr['n_classes'])

    start = n_train
    end = n_train + n_valid
    self.valid = Task('vector_classification',
                      name='valid',
                      x=ds._inputs[start:end].reshape(end-start, -1),
                      y=ds._labels[start:end],
                      n_classes=ds.descr['n_classes'])

    start = n_train + n_valid
    end = n_train + n_valid + n_test
    self.test = Task('vector_classification',
                     name='test',
                     x=ds._inputs[start:end].reshape(end-start, -1),
                     y=ds._labels[start:end],
                     n_classes=ds.descr['n_classes'])


def prepare_indexed_vector_classification(self):
    def task(name, idxs):
            return Task(
                'vector_classification',
                name=name,
                x=self.all_vectors[idxs],
                y=self.all_labels[idxs],
                n_classes=self.n_classes)

    self.train = task('sel', self.sel_idxs)
    self.test = task('tst', self.tst_idxs)


datasets = OrderedDict()
tasks = OrderedDict()

# Do NEVER modify this order as it determines the dataset ids
datasets["mnist"] = skdata.mnist.dataset.MNIST
tasks["mnist"] = skdata.mnist.view.OfficialVectorClassification
tasks["mnist"].prepare = \
    types.MethodType(prepare_indexed_vector_classification, None,
 tasks["mnist"])

datasets["larochelle_etal_2007_MNIST_BackgroundImages"] = \
    skdata.larochelle_etal_2007.dataset.MNIST_BackgroundImages
tasks["larochelle_etal_2007_MNIST_BackgroundImages"] = \
    skdata.larochelle_etal_2007.view.MNIST_BackgroundImages_VectorXV
tasks["larochelle_etal_2007_MNIST_BackgroundImages"].prepare = \
    types.MethodType(prepare, None, tasks["larochelle_etal_2007_MNIST_BackgroundImages"])

datasets["larochelle_etal_2007_MNIST_BackgroundRandom"] = \
    skdata.larochelle_etal_2007.dataset.MNIST_BackgroundRandom
tasks["larochelle_etal_2007_MNIST_BackgroundRandom"] = \
    skdata.larochelle_etal_2007.view.MNIST_BackgroundRandom_VectorXV
tasks["larochelle_etal_2007_MNIST_BackgroundRandom"].prepare = \
    types.MethodType(prepare, None, tasks["larochelle_etal_2007_MNIST_BackgroundRandom"])

datasets["larochelle_etal_2007_MNIST_Rotated"] = \
    skdata.larochelle_etal_2007.dataset.MNIST_Rotated
tasks["larochelle_etal_2007_MNIST_Rotated"] = \
    skdata.larochelle_etal_2007.view.MNIST_Rotated_VectorXV
tasks["larochelle_etal_2007_MNIST_Rotated"].prepare = \
    types.MethodType(prepare, None, tasks["larochelle_etal_2007_MNIST_Rotated"])

datasets["larochelle_etal_2007_MNIST_RotatedBackgroundImages"] = \
    skdata.larochelle_etal_2007.dataset.MNIST_RotatedBackgroundImages
tasks["larochelle_etal_2007_MNIST_RotatedBackgroundImages"] = \
    skdata.larochelle_etal_2007.view. MNIST_RotatedBackgroundImages_VectorXV
tasks["larochelle_etal_2007_MNIST_RotatedBackgroundImages"].prepare = \
    types.MethodType(prepare, None, tasks["larochelle_etal_2007_MNIST_RotatedBackgroundImages"])

datasets["larochelle_etal_2007_MNIST_Noise1"] = \
    skdata.larochelle_etal_2007.dataset.MNIST_Noise1
tasks["larochelle_etal_2007_MNIST_Noise1"] = \
    skdata.larochelle_etal_2007.view.MNIST_Noise1_VectorXV
tasks["larochelle_etal_2007_MNIST_Noise1"].prepare = \
    types.MethodType(prepare, None, tasks["larochelle_etal_2007_MNIST_Noise1"])

datasets["larochelle_etal_2007_MNIST_Noise2"] = \
    skdata.larochelle_etal_2007.dataset.MNIST_Noise2
tasks["larochelle_etal_2007_MNIST_Noise2"] = \
    skdata.larochelle_etal_2007.view.MNIST_Noise1_VectorXV
tasks["larochelle_etal_2007_MNIST_Noise2"].prepare = \
    types.MethodType(prepare, None, tasks["larochelle_etal_2007_MNIST_Noise2"])

datasets["larochelle_etal_2007_MNIST_Noise3"] = \
    skdata.larochelle_etal_2007.dataset.MNIST_Noise3
tasks["larochelle_etal_2007_MNIST_Noise3"] = \
    skdata.larochelle_etal_2007.view.MNIST_Noise3_VectorXV
tasks["larochelle_etal_2007_MNIST_Noise3"].prepare = \
    types.MethodType(prepare, None, tasks["larochelle_etal_2007_MNIST_Noise3"])

datasets["larochelle_etal_2007_MNIST_Noise4"] = \
    skdata.larochelle_etal_2007.dataset.MNIST_Noise4
tasks["larochelle_etal_2007_MNIST_Noise4"] = \
    skdata.larochelle_etal_2007.view.MNIST_Noise4_VectorXV
tasks["larochelle_etal_2007_MNIST_Noise4"].prepare = \
    types.MethodType(prepare, None, tasks["larochelle_etal_2007_MNIST_Noise4"])

datasets["larochelle_etal_2007_MNIST_Noise5"] = \
    skdata.larochelle_etal_2007.dataset.MNIST_Noise5
tasks["larochelle_etal_2007_MNIST_Noise5"] = \
    skdata.larochelle_etal_2007.view.MNIST_Noise5_VectorXV
tasks["larochelle_etal_2007_MNIST_Noise5"].prepare = \
    types.MethodType(prepare, None, tasks["larochelle_etal_2007_MNIST_Noise5"])

datasets["larochelle_etal_2007_MNIST_Noise6"] = \
    skdata.larochelle_etal_2007.dataset.MNIST_Noise6
tasks["larochelle_etal_2007_MNIST_Noise6"] = \
    skdata.larochelle_etal_2007.view.MNIST_Noise6_VectorXV
tasks["larochelle_etal_2007_MNIST_Noise6"].prepare = \
    types.MethodType(prepare, None, tasks["larochelle_etal_2007_MNIST_Noise6"])

datasets["larochelle_etal_2007_Rectangles"] = \
    skdata.larochelle_etal_2007.dataset.Rectangles
tasks["larochelle_etal_2007_Rectangles"] = \
    skdata.larochelle_etal_2007.view.RectanglesVectorXV
tasks["larochelle_etal_2007_Rectangles"].prepare = \
    types.MethodType(prepare, None, tasks["larochelle_etal_2007_Rectangles"])

datasets["larochelle_etal_2007_RectanglesImages"] = \
    skdata.larochelle_etal_2007.dataset.RectanglesImages
tasks["larochelle_etal_2007_RectanglesImages"] = \
    skdata.larochelle_etal_2007.view.RectanglesImagesVectorXV
tasks["larochelle_etal_2007_RectanglesImages"].prepare = \
    types.MethodType(prepare, None, tasks["larochelle_etal_2007_RectanglesImages"])

datasets["larochelle_etal_2007_Convex"] = \
    skdata.larochelle_etal_2007.dataset.Convex
tasks["larochelle_etal_2007_Convex"] = \
    skdata.larochelle_etal_2007.view.ConvexVectorXV
tasks["larochelle_etal_2007_Convex"].prepare = \
    types.MethodType(prepare, None, tasks["larochelle_etal_2007_Convex"])

datasets["cifar10"] = skdata.cifar10.dataset.CIFAR10
tasks["cifar10"] = skdata.cifar10.views.OfficialVectorClassificationTask


# SVHN has too many dimensions
# datasets["svhn"] = skdata.svhn.dataset.CroppedDigits
# tasks["svhn"] = skdata.svhn.view.CroppedDigitsView2


"""
def get_local_directory():
    return skdata_.data_home.get_data_home()


def set_local_directory(newpath):
    return skdata_.data_home.set_data_home(newpath)


def get_local_datasets():
    local_datasets = OrderedDict()
    for i, dataset in local_datasets:
        local_datasets[i] = dataset
    return local_datasets


def get_local_dataset(name):
    try:
        return datasets[name]()
    except KeyError:
        print "Dataset not known"


def get_remote_datasets():
    pass


def list_local_datasets():
    pass


def show_remote_datasets():
    pass


def show_only_remote_datasets():
    pass
"""


# TODO split this function into small and testable chunks
def convert_dataset_to_openml_dataset(name):
    dataset = datasets[name]()
    if name in set(["cifar10", "mnist"]) or \
            isinstance(dataset, skdata.larochelle_etal_2007.dataset.BaseL2007):
        dataset.fetch(True)
    elif name in set(["iris", "diabetes", "digits"]):
        pass
    else:
        dataset.fetch()

    task = tasks[name]()
    if hasattr(task, "prepare"):
        task.prepare()

    if name == "mnist":
        task.train.x = np.array(task.train.x.reshape((-1, 784)), dtype=np.float32)
        task.test.x = np.array(task.test.x.reshape((-1, 784)), dtype=np.float32)
    elif name == "larochelle_etal_2007_Convex":
        pass
    elif task.train.x.dtype != np.float32:
        print task.train.x, type(task.train.x), task.train.x.dtype
        raise Exception

    local_directory = pyMetaLearn.openml.manage_openml_data.get_local_directory()
    split_dir = os.path.join(local_directory, "splits")
    dataset_dir = os.path.join(local_directory, "datasets")

    # Gather all information for a dataset
    source_community = "skdata"
    did = 1000000 + datasets.keys().index(name)
    name = name
    version = 1.0
    description = "Downloaded via skdata. Please refer to skdata for the " \
                  "exact description of the datasets."
    format = "arff"
    url = None
    md5_checksum = None
    local_directory = local_directory
    default_target_attribute = None

    ds = OpenMLDataset(source_community, did, name, version, description,
                     format, url, md5_checksum, local_directory,
                     default_target_attribute)

    # Write the dataset as as .pkl file
    dataset_description_filename = os.path.join(dataset_dir, 'did' + str(did) +
                                                 ".pkl")
    with open(dataset_description_filename, "w") as fh:
        cPickle.dump({'source_community': 'skdata',
                      'id': did,
                      'name': name,
                      'version': version,
                      'description': description,
                      'format': format,
                      'url': url,
                      'md5_checksum': md5_checksum,
                      'local_directory': local_directory,
                      'default_target_attribute': default_target_attribute}, fh)

    # Convert the dataset to the ARFF format
    ds_as_arff = {}
    ds_as_arff['relation'] = name
    ds_as_arff['attributes'] = [('%d' % idx, 'NUMERIC') for idx, dimension in
                                enumerate(task.train.x[0])]
    ds_as_arff['attributes'].append(('class', 'INTEGER'))

    # Feed one numpy array per pattern
    if hasattr(task, 'test'):
        data = []
        for line in np.hstack((task.train.x, task.train.y.reshape((-1, 1)))):
            data.append(line)
        if hasattr(task, 'valid'):
            for line in np.hstack((task.valid.x, task.valid.y.reshape((-1, 1)))):
                data.append(line)
        for line in np.hstack((task.test.x, task.test.y.reshape((-1, 1)))):
            data.append(line)
    else:
        raise NotImplementedError()
    ds_as_arff['data'] = data
    dataset_filename = os.path.join(dataset_dir, 'did' + str(did) +
                                                 "_" + name + ".arff.gz")
    # According to http://tukaani.org/lzma/benchmarks.html, decompression
    # time does not depend on the compression level
    if not os.path.exists(dataset_filename):
        with gzip.open(dataset_filename, "w") as fh:
            arff.dump(ds_as_arff, fh)

    dataset_identifier = "did_%d_%s_" % (did, name)
    split_identifier = "original_split.arff"
    split_file = os.path.join(split_dir, dataset_identifier + split_identifier)
    cv_split_identifier = "generated:10foldcv_inside_original_split.arff"
    cv_split_file = os.path.join(split_dir, dataset_identifier + cv_split_identifier)

    t = OpenMLTask(did, "Supervised Classification", did, 1, "crossvalidation with original split",
                   None, {}, "predictive_accuracy", split_file, cv_split_file)

    # Create the split files
    # Use np.arange because later on we need the array to actually index the
    # dataset array
    # Case of a pre-defined test split:
    if hasattr(task, "test"):
        # Case of a pre-defined validation split (which we ignore and instead
        #  perform a 10fold crossvalidation
        if hasattr(task, 'valid'):
            train_indices = np.arange(len(task.train.x) + len(task.valid.x))

        else:
            train_indices = np.arange(len(task.train.x))
        test_indices = np.arange(len(task.test.x))

        splits = {}
        splits['description'] = "Original dataset split"
        splits['relation'] = "dataset_%d_%s_splits" % (did, name)
        splits['attributes'] = [('type', ('TRAIN', 'TEST')),
                                ('rowid', 'INTEGER'),
                                ('repeat', 'INTEGER'),
                                ('fold', 'INTEGER')]
        splits['data'] = [('TRAIN',idx,0,0) for idx in train_indices]
        offset = len(splits['data'])
        splits['data'].extend([('TEST',offset+idx,0,0) for idx in test_indices])
        with open(split_file, "w") as fh:
            arff.dump(splits, fh)
        del splits

        # These two are not needed any more
        del dataset
        del task

        valid_splits = {}
        valid_splits['description'] = "Preliminary crossvalidation splits " \
                                      "generated by Matthias Feurer"
        valid_splits['relation'] = "dataset_%d_%s_splits" % (did, name)
        valid_splits['attributes'] = [('type', ('TRAIN', 'TEST')),
                                      ('rowid', 'INTEGER'),
                                      ('repeat', 'INTEGER'),
                                      ('fold', 'INTEGER')]
        valid_splits['data'] = []

        X, Y = ds.get_npy(target='class', scaling='scale')
        for fold in range(10):
            valid_train_split, valid_test_split = \
                t._get_fold(X[train_indices], Y[train_indices],
                            fold=fold, folds=10, shuffle=True)
            valid_splits['data'].extend([('TRAIN',idx,0,fold) for idx in valid_train_split])
            valid_splits['data'].extend([('TEST',idx,0,fold) for idx in valid_test_split])
        with open(cv_split_file, "w") as fh:
            arff.dump(valid_splits, fh)
        del valid_splits

    else:
        raise NotImplementedError()


if __name__ == "__main__":
    for dataset in reversed(datasets):

        print "###"
        print dataset
        ds = datasets[dataset]()
        if dataset in set(["cifar10", "mnist"]) or \
                isinstance(ds, skdata.larochelle_etal_2007.dataset.BaseL2007):
            ds.fetch(True)
        elif dataset in set(["iris", "diabetes", "digits"]):
            pass
        else:
            ds.fetch()

        task = tasks[dataset]()
        if hasattr(task, "prepare"):
            task.prepare()

        if dataset == "mnist":
            task.train.x = np.array(task.train.x.reshape((-1, 784)), dtype=np.float32)
            task.test.x = np.array(task.test.x.reshape((-1, 784)), dtype=np.float32)
        elif dataset == "convex":
            pass
        elif task.train.x.dtype != np.float32:
            print task.train.x, type(task.train.x), task.train.x.dtype
            raise Exception

        print vars(task)
        print task.train.x.shape, task.valid.x.shape, task.test.x.shape

        starttime = time.time()
        random_state = sklearn.utils.check_random_state(42)
        fn = sklearn.ensemble.RandomForestClassifier(random_state=random_state)
        scaler = sklearn.preprocessing.MinMaxScaler(copy=True).fit(task.train.x)
        x_train = task.train.x.copy()
        x_test = task.test.x.copy()
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        fn.fit(x_train, task.train.y)
        prediction = fn.predict(x_test)
        print "RF", sklearn.metrics.accuracy_score(task.test.y, prediction)
        print time.time() - starttime

        """
        starttime = time.time()
        random_state = sklearn.utils.check_random_state(42)
        fn = sklearn.svm.SVC(cache_size=2000, random_state=random_state)
        scaler = sklearn.preprocessing.MinMaxScaler(copy=True).fit(task.train.x)
        x_train = task.train.x.copy()
        x_test = task.test.x.copy()
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        fn.fit(x_train, task.train.y)
        prediction = fn.predict(x_test)
        print "SVM", sklearn.metrics.accuracy_score(task.test.y, prediction)
        print time.time() - starttime
        """

        del task
