from collections import defaultdict
import numpy as np

import pyMetaLearn.openml.manage_openml_data
from pyMetaLearn.openml.openml_dataset import OpenMLDataset
from pyMetaLearn.openml.openml_task import OpenMLTask

import sklearn.ensemble

################################################################################
# Iterate all OpenML supervised classification tasks
tasks = pyMetaLearn.openml.manage_openml_data.get_remote_tasks()
downloaded_tasks = []

did_to_targets = defaultdict(set)

################################################################################
# Download all these tasks

for tid in tasks:
    task = pyMetaLearn.openml.manage_openml_data.download_task(tid)
    pyMetaLearn.openml.manage_openml_data.download(int(task.dataset_id))
    downloaded_tasks.append(task)
    did_to_targets[int(task.dataset_id)].add(task.target_feature)


################################################################################
# We want to use only the datasets and create some tasks ourselves

# TODO: compile a list of datasets I want to use
# TODO: If there are different targets for a dataset, figure out a way to
# perform more than one experiment
print did_to_targets

datasets = pyMetaLearn.openml.manage_openml_data.get_local_datasets()

print
print "Available datasets:"
for did in datasets:
    dataset = OpenMLDataset.from_xml_file(datasets[did])
    print did, datasets[did], dataset._name

to_use = list()
for r in [range(1, 17), (18,), range(20, 25), range(26, 38), range(39, 52), \
        range(53, 57), range(58, 63)]:
    to_use.extend(r)

print "Datasets which will be used", to_use
print "This are %d datasets." % len(to_use)
print

for did in to_use:
    if len(did_to_targets[did]) != 1:
        raise NotImplementedError()
    # print did, did_to_targets[did]

    task = OpenMLTask(100000 + did, "Supervised Classification", did,
                did_to_targets[did].pop().lower(),
                "crossvalidation with crossvalidation holdout", None,
                {"stratified_sampling": "true", "test_folds": 3,
                "test_fold": 0}, "predictive_accuracy")

    algo = sklearn.ensemble.RandomForestClassifier(n_jobs=4)
    vals = []
    print "DID", did,
    for i in range(10):
        vals.append(1. - task.partial_evaluate(algo, 0, 10))
    print np.mean(vals)

