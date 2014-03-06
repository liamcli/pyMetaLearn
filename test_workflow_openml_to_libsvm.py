import os
import unittest
import shutil

import openml.manage_openml_data as manage_openml_data
import metafeatures.metafeatures as metafeatures
import metafeatures.plot_metafeatures as pmf

class TestWorkflow(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = ".test_workflow_openml_to_libsvm"
        os.mkdir(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_workflow(self):
        """Test the workflow of downloading a dataset from OpenML to running
        a gridsearch experiment with the LibSVM through HPOlib on that dataset:
        1. Create a directory to perform experiments in (setUp method)
        2. Download a dataset from the openml server
        3. Retrieve the processed dataset and calculate meta-features
        4. Plot the meta-features
        5. Create an experiment directory
        6. Run a small gridsearch on the new experiment
        """

        datasets = manage_openml_data.get_remote_datasets(names=False)
        # TODO: Why doesn't the API accept a single integer?
        dataset = manage_openml_data.download(self.tmp_dir, (datasets[0],))
        self.assertEqual(len(dataset), 1)
        dataset = dataset[0]
        self.assertEqual(dataset._name, "anneal")

        # TODO: replace with get_processed_files()
        # TODO: Why does the function want the dataset object?
        dataframe = dataset.convert_arff_structure_to_pandas(dataset
                                                        .get_unprocessed_files())
        # TODO: Why doesn't the dataset calculate the metafeatures?
        # TODO: add the target attribute to the dataset class and return X
        # and Y according to that
        class_ = dataframe.keys()[-1]
        attributes = dataframe.keys()[0:-1]
        X = dataframe[attributes]
        Y = dataframe[class_]
        mfs = metafeatures.calculate_all_metafeatures(X, Y)
        self.assertEqual(len(mfs), 18)

        # Pretty bad idea to assume a directory called ../figures
        pmf.plot_metafeatures(self.tmp_dir)





