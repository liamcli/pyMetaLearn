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
"""

# TODO: add PIL to dependencies

import skdata.cifar10
import skdata.iris
import skdata.kaggle_facial_expression
import skdata.larochelle_etal_2007.dataset
import skdata.mnist
import skdata.pubfig.dataset
import skdata.svhn
import skdata.brodatz
import skdata.caltech
import skdata.diabetes
import skdata.digits
import skdata.pubfig83

datasets = dict()
datasets["cifar10"] = skdata.cifar10.dataset.CIFAR10()
datasets["iris"] = skdata.iris.dataset.Iris()
datasets["kaggle_facial_expression"] = skdata.kaggle_facial_expression.dataset.KaggleFacialExpression()
datasets["larochelle_etal_2007_MNIST_BackgroundImages"] = skdata.larochelle_etal_2007.dataset.MNIST_BackgroundImages()
datasets["larochelle_etal_2007_MNIST_BackgroundRandom"] = skdata.larochelle_etal_2007.dataset.MNIST_BackgroundRandom()
datasets["larochelle_etal_2007_MNIST_Rotated"] = skdata.larochelle_etal_2007.dataset.MNIST_Rotated()
datasets["larochelle_etal_2007_MNIST_RotatedBackgroundImages"] = \
    skdata.larochelle_etal_2007.dataset.MNIST_RotatedBackgroundImages()
datasets["larochelle_etal_2007_MNIST_Noise1"] = skdata.larochelle_etal_2007.dataset.MNIST_Noise1()
datasets["larochelle_etal_2007_MNIST_Noise2"] = skdata.larochelle_etal_2007.dataset.MNIST_Noise2()
datasets["larochelle_etal_2007_MNIST_Noise3"] = skdata.larochelle_etal_2007.dataset.MNIST_Noise3()
datasets["larochelle_etal_2007_MNIST_Noise4"] = skdata.larochelle_etal_2007.dataset.MNIST_Noise4()
datasets["larochelle_etal_2007_MNIST_Noise5"] = skdata.larochelle_etal_2007.dataset.MNIST_Noise5()
datasets["larochelle_etal_2007_MNIST_Noise6"] = skdata.larochelle_etal_2007.dataset.MNIST_Noise6()
datasets["larochelle_etal_2007_Rectangles"] = skdata.larochelle_etal_2007.dataset.Rectangles()
datasets["larochelle_etal_2007_RectanglesImages"] = skdata.larochelle_etal_2007.dataset.RectanglesImages()
datasets["larochelle_etal_2007_Convex"] = skdata.larochelle_etal_2007.dataset.Convex()
datasets["mnist"] = skdata.mnist.dataset.MNIST()
datasets["svhn"] = skdata.svhn.dataset.CroppedDigits()
datasets["brodatz"] = skdata.brodatz.Brodatz()
datasets["caltech101"] = skdata.caltech.Caltech101()
datasets["caltech256"] = skdata.caltech.Caltech256()
datasets["diabetes"] = skdata.diabetes.Diabetes()
datasets["digits"] = skdata.digits.Digits()

for dataset in datasets:
    print dataset
    if dataset == "cifar10" or isinstance(datasets[dataset], skdata.larochelle_etal_2007.dataset.BaseL2007):
        datasets[dataset].fetch(True)
    else:
        datasets[dataset].fetch()
    try:
        datasets[dataset].descr
    except:
        pass
    try:
        vars(datasets[dataset])
    except:
        pass
