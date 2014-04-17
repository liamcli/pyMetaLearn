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
from collections import OrderedDict
import types
import time


import skdata
import sys
print sys.path
print skdata.__file__
import skdata.base
from skdata.base import Task

import sklearn.utils
import sklearn.ensemble.forest as RandomForest
import sklearn.svm
import sklearn.metrics

# TODO: add PIL to dependencies
import skdata.cifar10
import skdata.iris
import skdata.kaggle_facial_expression
import skdata.larochelle_etal_2007.dataset
import skdata.larochelle_etal_2007.view
import skdata.mnist
import skdata.pubfig.dataset
import skdata.svhn
import skdata.brodatz
import skdata.caltech
import skdata.diabetes
import skdata.digits
import skdata.pubfig83


def prepare(self):
    """
    This is modification from skdata/larochelle_et_al_2007/view.py and will be
    injected in there instead of the original protocol
    """
    print "Preparing"

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

datasets["cifar10"] = skdata.cifar10.dataset.CIFAR10
tasks["cifar10"] = skdata.cifar10.views.OfficialVectorClassificationTask

"""
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

datasets["mnist"] = skdata.mnist.dataset.MNIST
tasks["mnist"] = skdata.mnist.view.OfficialVectorClassification
tasks["mnist"].prepare = \
    types.MethodType(prepare_indexed_vector_classification, None, tasks["mnist"])
"""

# SVHN has too many dimensions
# datasets["svhn"] = skdata.svhn.dataset.CroppedDigits
# tasks["svhn"] = skdata.svhn.view.CroppedDigitsView2



def get_local_directory():
    return skdata.data_home.get_data_home()


def set_local_directory(newpath):
    return skdata.data_home.set_data_home(newpath)


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

    starttime = time.time()

    random_state = sklearn.utils.check_random_state(42)
    fn = sklearn.svm.SVC(cache_size=1000, random_state=random_state)
    #fn = RandomForest.RandomForestClassifier(n_estimators=100,
    #    criterion="entropy", n_jobs=1, random_state=random_state)
    fn.fit(task.train.x, task.train.y)
    prediction = fn.predict(task.test.x)
    print sklearn.metrics.accuracy_score(task.test.y, prediction)
    print time.time() - starttime

    del task
"""