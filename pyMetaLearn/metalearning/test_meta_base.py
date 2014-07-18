from collections import OrderedDict
import os
import StringIO
import unittest

import pandas as pd

import pyMetaLearn.metalearning.meta_base
data_dir = os.path.dirname(pyMetaLearn.metalearning.meta_base.__file__)
data_dir = os.path.join(data_dir, 'test_meta_base_data')

import pyMetaLearn.openml.manage_openml_data
from pyMetaLearn.metalearning.meta_base import MetaBase, Run


class MetaBaseTest(unittest.TestCase):
    def setUp(self):
        self.cwd = os.getcwd()
        pyMetaLearn.openml.manage_openml_data.set_local_directory(data_dir)
        os.chdir(data_dir)

        experiments_list = StringIO.StringIO()
        experiments_list.write(
            os.path.join(data_dir, 'base_runs', 'sequential_did_2.pkl'))
        experiments_list.write("\n")
        experiments_list.write(
            os.path.join(data_dir, 'base_runs', 'sequential_did_3.pkl'))
        experiments_list.write("\n")
        experiments_list.write(
            os.path.join(data_dir, 'base_runs', 'sequential_did_4.pkl'))
        experiments_list.write("\n")
        experiments_list.seek(0)

        task_list = StringIO.StringIO()
        task_list.write(os.path.join(data_dir, 'tasks', 'did_2.pkl'))
        task_list.write("\n")
        task_list.write(os.path.join(data_dir, 'tasks', 'did_3.pkl'))
        task_list.write("\n")
        task_list.write(os.path.join(data_dir, 'tasks', 'did_4.pkl'))
        task_list.write("\n")
        task_list.seek(0)

        self.base = MetaBase(task_list, experiments_list)

    def tearDown(self):
        os.chdir(self.cwd)

    def test_get_dataset(self):
        ds = self.base.get_dataset('anneal.ORIG')
        self.assertEqual(ds._name, 'anneal.ORIG')

    def test_get_datasets(self):
        datasets = self.base.get_datasets()
        self.assertEqual(type(datasets), OrderedDict)
        self.assertEqual(len(datasets), 3)

    def test_get_runs(self):
        runs = self.base.get_runs('anneal.ORIG')
        self.assertEqual(1686, len(runs))
        for run in runs:
            self.assertEqual(Run, type(run))

    def test_get_cv_runs(self):
        runs = self.base.get_cv_runs('anneal.ORIG')
        self.assertEqual(1686, len(runs))


    def test_get_metafeatures_as_pandas(self):
        mf = self.base.get_metafeatures_as_pandas('kr-vs-kp')
        self.assertEqual(type(mf), pd.Series)
        self.assertEqual(mf.name, 'kr-vs-kp')
        self.assertEqual(mf.loc['number_of_instances'], 3196)

    def test_get_all_metafeatures_as_pandas(self):
        mf = self.base.get_all_metafeatures_as_pandas()
        self.assertEqual(type(mf), pd.DataFrame)
        self.assertEqual(3, mf.shape[0])
        self.assertEqual(mf.index[0], 'anneal.ORIG')
        self.assertEqual(mf.loc['anneal.ORIG', 'number_of_instances'], 898)
        self.assertEqual(mf.index[1], 'kr-vs-kp')

    def test_get_train_metafeatures_as_pandas(self):
        mf = self.base.get_train_metafeatures_as_pandas('kr-vs-kp')
        self.assertEqual(type(mf), pd.Series)
        self.assertEqual(mf.name, 'kr-vs-kp')
        self.assertEqual(mf.loc['number_of_instances'], 2130)

    def test_get_all_train_metafeatures_as_pandas(self):
        print os.getcwd()
        mf = self.base.get_all_train_metafeatures_as_pandas()
        self.assertEqual(type(mf), pd.DataFrame)
        self.assertEqual(3, mf.shape[0])
        self.assertEqual(mf.index[0], 'anneal.ORIG')
        self.assertEqual(mf.loc['anneal.ORIG', 'number_of_instances'], 598)
        self.assertEqual(mf.index[1], 'kr-vs-kp')

    """
    def get_cv_metafeatures_as_pandas(...

    def get_all_cv_metafeatures_as_pandas(...
    """

    def test_read_experiment_pickle(self):
        self.base.read_experiment_pickle()

    def test_read_folds_from_experiment_pickle(self):
        self.base.read_folds_from_experiment_pickle()
