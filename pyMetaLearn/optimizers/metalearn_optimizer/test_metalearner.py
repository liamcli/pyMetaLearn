from collections import OrderedDict
import StringIO
import numpy as np
import os
import unittest

import pandas as pd

import pyMetaLearn.metalearning.meta_base
data_dir = os.path.dirname(pyMetaLearn.metalearning.meta_base.__file__)
data_dir = os.path.join(data_dir, 'test_meta_base_data')

import pyMetaLearn.openml.manage_openml_data
from pyMetaLearn.metalearning.meta_base import Run, MetaBase
import pyMetaLearn.optimizers.metalearn_optimizer.metalearner as metalearner


class MetaLearnerTest(unittest.TestCase):
    def setUp(self):
        self.anneal = pd.Series({"number_of_instances": 898., "number_of_classes": 5.,
                            "number_of_features": 38.}, name="anneal")
        self.krvskp = pd.Series({"number_of_instances": 3196., "number_of_classes":
                            2., "number_of_features": 36.}, name="krvskp")
        self.labor = pd.Series({"number_of_instances": 57., "number_of_classes":
                           2., "number_of_features": 16.}, name="labor")
        self.runs = {'anneal': [Run({'x': 0}, 0.1), Run({'x': 1}, 0.5), Run({'x': 2}, 0.7)],
                'krvskp': [Run({'x': 0}, 0.5), Run({'x': 1}, 0.1), Run({'x': 2}, 0.7)],
                'labor': [Run({'x': 0}, 0.5), Run({'x': 1}, 0.7), Run({'x': 2}, 0.1)]}

        self.cwd = os.getcwd()
        openml_dir = pyMetaLearn.openml.manage_openml_data.set_local_directory(
            data_dir)
        os.chdir(openml_dir)

        task_file = os.path.join(openml_dir, 'tasks', 'did_2.pkl')
        task_list = StringIO.StringIO()
        task_list.write(os.path.join(openml_dir, 'tasks', 'did_2.pkl'))
        task_list.write('\n')
        task_list.write(os.path.join(openml_dir, 'tasks', 'did_3.pkl'))
        task_list.write("\n")
        task_list.write(os.path.join(openml_dir, 'tasks', 'did_4.pkl'))
        task_list.write("\n")
        task_list.seek(0)
        experiments_list = StringIO.StringIO()
        experiments_list.write('\n')
        experiments_list.write(
            os.path.join(openml_dir, 'base_runs', 'sequential_did_3.pkl'))
        experiments_list.write('\n')
        experiments_list.write(
            os.path.join(openml_dir, 'base_runs', 'sequential_did_4.pkl'))
        experiments_list.write("\n")
        experiments_list.seek(0)

        self.meta_optimizer = metalearner.MetaLearningOptimizer(task_file,
            task_list, experiments_list, openml_dir)

    def tearDown(self):
        os.chdir(self.cwd)

    def test_perform_sequential_optimization(self):
        # TODO: this is only a smoke test!
        def dummy_function(params):
            return params
        ret = self.meta_optimizer.perform_sequential_optimization(
            target_algorithm=dummy_function, evaluation_budget=2)
        self.assertEqual(type(ret), OrderedDict)
        with self.assertRaises(StopIteration):
            self.meta_optimizer.perform_sequential_optimization(dummy_function)


    def test_metalearning_suggest_all(self):
        ret = self.meta_optimizer.metalearning_suggest_all()
        self.assertEqual(2, len(ret))
        self.assertEqual(OrderedDict([('-classifier', 'random_forest'),
                                      ('-preprocessing', 'None'),
                                      ('-random_forest:criterion', 'entropy'),
                                      ('-random_forest:max_features', '5'),
                                      ('-random_forest:min_samples_split', '0')]),
                         ret[0])
        # There is no test for exclude_double_configuration as it's not present
        # in the test data

    def test_metalearning_suggest(self):
        ret = self.meta_optimizer.metalearning_suggest(list())
        self.assertEqual(type(ret), OrderedDict)
        self.assertEqual(OrderedDict([('-classifier', 'random_forest'),
                                      ('-preprocessing', 'None'),
                                      ('-random_forest:criterion', 'entropy'),
                                      ('-random_forest:max_features', '5'),
                                      ('-random_forest:min_samples_split', '0')]),
                         ret)
        ret2 = self.meta_optimizer.metalearning_suggest([Run(ret, 1)])
        self.assertEqual(type(ret2), OrderedDict)
        self.assertEqual(OrderedDict([('-classifier', 'libsvm_svc'),
                                      ('-libsvm_svc:C', 0.03125),
                                      ('-libsvm_svc:gamma', 3.0517578125e-05),
                                      ('-preprocessing', 'None')]), ret2)


    def test_learn(self):
        # Test only some special cases which are probably not yet handled
        # like the metafeatures to eliminate and the random forest
        # hyperparameters
        self.meta_optimizer._learn()

    def test_get_metafeatures(self):
        metafeatures, all_other_metafeatures = \
            self.meta_optimizer._get_metafeatures()
        self.assertEqual(type(metafeatures), pd.Series)
        self.assertEqual(type(all_other_metafeatures), pd.DataFrame)
        self.assertEqual('anneal.ORIG', metafeatures.name)
        self.assertLess(2, metafeatures.shape[0])
        self.meta_optimizer.use_features = ['number_of_classes']
        metafeatures, all_other_metafeatures = \
            self.meta_optimizer._get_metafeatures()
        self.assertGreater(2, metafeatures.shape[0])


    def test_read_task_list(self):
        task_list_file = StringIO.StringIO()
        task_list_file.write('a\nb\nc\nd\n')
        task_list_file.seek(0)
        task_list = self.meta_optimizer.read_task_list(task_list_file)
        self.assertEqual(4, len(task_list))

        task_list_file = StringIO.StringIO()
        task_list_file.write('a\n\nc\nd\n')
        task_list_file.seek(0)
        self.assertRaisesRegexp(ValueError, 'Blank lines in the task list are not supported.',
                                self.meta_optimizer.read_task_list,
                                task_list_file)

    def test_read_experiments_list(self):
        experiments_list_file = StringIO.StringIO()
        experiments_list_file.write('a\nb\n\nc d\n')
        experiments_list_file.seek(0)
        experiments_list = self.meta_optimizer.read_experiments_list(
            experiments_list_file)
        self.assertEqual(4, len(experiments_list))
        self.assertEqual(2, len(experiments_list[3]))

    def test_split_metafeature_array(self):
        metafeatures = pd.DataFrame([self.anneal, self.krvskp, self.labor])

        ds_metafeatures, other_metafeatures = self.meta_optimizer. \
            _split_metafeature_array("krvskp", metafeatures)
        self.assertIsInstance(ds_metafeatures, pd.Series)
        self.assertEqual(len(other_metafeatures.index), 2)


if __name__ == "__main__":
    unittest.main()

