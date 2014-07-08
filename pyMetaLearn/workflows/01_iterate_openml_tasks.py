from collections import defaultdict
import cPickle
import os

import pyMetaLearn.openml.manage_openml_data
from pyMetaLearn.openml.openml_dataset import OpenMLDataset
from pyMetaLearn.openml.openml_task import OpenMLTask


pyMetaLearn.openml.manage_openml_data.set_local_directory(
    "/home/feurerm/thesis/datasets/openml/")
print "Starting to iterate datasets..."
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
# Issues with datasets
# 1 - is the same as 2
# 17, 19 target attribute contains missing values
# 25, 27 contains a semi-index
# 38 does not exist
# 45 is only the test set
# 46 contains indices
# 47 is only the train set for 45
# 52 contains to less data
# 57 does not exist
# 62 contains some semi-index
# 63-69 does not exist
# 70-78 are generated datasets
# 79-114 don't exist
# 115-162 are generated datasets
# 163 has a wrong class attribute
# 164 has a unique identifier
# 165-170 have wrong targets + they are split into train and test...
# 172 has wrong target + too less data
# 173, 174 have too less target examples of target classes other than zero
# 175, 176 have the wrong class attribute
# 179, 180 take too long
# 184 takes too long
# 185 has an index
# 187 does not exist
for r in [range(2, 17), (18,), range(20, 25), range(26, 27), range(28, 38), \
        range(39, 45), range(48, 52), range(53, 57), range(58, 63), (171,),
        range(181, 183), (186,), (188,)]:
    to_use.extend(r)

print "Datasets which will be used", to_use
print "This are %d datasets." % len(to_use)
print

local_directory = pyMetaLearn.openml.manage_openml_data.get_local_directory()
custom_tasks_dir = os.path.join(local_directory, "custom_tasks")
try:
    os.mkdir(custom_tasks_dir)
except:
    pass

for did in to_use:
    if len(did_to_targets[did]) != 1:
        print did, did_to_targets[did]
        raise NotImplementedError()

    task_properties = {"task_id": did,
                       "task_type": "Supervised Classification",
                       "data_set_id": did,
                       "target_feature": did_to_targets[did].pop(),
                       "estimation_procudure_type": "crossvalidation with crossvalidation holdout",
                       "data_splits_url": None,
                       "estimation_parameters": {"stratified_sampling": "true", "test_folds": 3,
                                                 "test_fold": 0},
                       "evaluation_measure": "predictive_accuracy"}

    task_file = os.path.join(custom_tasks_dir, "did_%d.pkl" %
        task_properties["task_id"])

    with open(task_file, "w") as fh:
        cPickle.dump(task_properties, fh)