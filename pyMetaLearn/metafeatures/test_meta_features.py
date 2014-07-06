from unittest import TestCase
import unittest
import os

import numpy as np
import pyMetaLearn
from pyMetaLearn.openml.openml_dataset import OpenMLDataset
import pyMetaLearn.metafeatures.metafeatures as meta_features
import pyMetaLearn.openml.manage_openml_data
import arff

__author__ = 'feurerm'


class TestMetaFeatures(TestCase):
    def setUp(self):
        self.openml_dir = os.path.join(os.path.dirname(pyMetaLearn.__file__),
                                  "openml")
        self.test_cache_dir = os.path.join(self.openml_dir, "test_cache")
        fh = open(os.path.join(self.openml_dir,
                               "test_cache/datasets/did1_anneal.arff"))
        self.arff_object = arff.load(fh)
        fh.close()
        pyMetaLearn.openml.manage_openml_data.set_local_directory(
            self.test_cache_dir)
        self.ds = OpenMLDataset("OpenML", 1, "anneal", None, None, "arff",
                                None, None, self.test_cache_dir, None, True)
        X, Y = self.ds.get_pandas()
        self.X = X
        self.Y = Y
        Xnp, Ynp = self.ds.get_npy(scaling="scale")
        self.Xnp = Xnp
        self.Ynp = Ynp
        self.mf = meta_features.metafeatures
        self.helpers = meta_features.helper_functions

        # Precompute some helper functions
        self.helpers.set_value("PCA", self.helpers["PCA"](self.Xnp,
                                                          self.Ynp), 1)
        self.helpers.set_value("missing_values", self.helpers[
            "missing_values"](self.X, self.Y), 1)
        self.helpers.set_value("num_symbols", self.helpers["num_symbols"](
            self.X, self.Y), 1)
        self.helpers.set_value("class_occurences",
                               self.helpers["class_occurences"](self.X, self.Y), 1)

    def tearDown(self):
        os.remove(os.path.join(self.test_cache_dir, "datasets", "did1_x.df"))
        os.remove(os.path.join(self.test_cache_dir, "datasets", "did1_y.df"))
        os.remove(os.path.join(self.test_cache_dir, "datasets", "did1_x.npy"))
        os.remove(os.path.join(self.test_cache_dir, "datasets", "did1_y.npy"))

    def test_number_of_instance(self):
        mf = self.mf["number_of_instances"](self.X, self.Y)
        self.assertEqual(mf, 898)
        self.assertIsInstance(mf, float)

    def test_number_of_classes(self):
        mf = self.mf["number_of_classes"](self.X, self.Y)
        self.assertEqual(mf, 5)
        self.assertIsInstance(mf, float)

    def test_number_of_features(self):
        mf = self.mf["number_of_features"](self.X, self.Y)
        self.assertEqual(mf, 38)
        self.assertIsInstance(mf, float)

    def test_number_of_Instances_with_missing_values(self):
        mf = self.mf["number_of_Instances_with_missing_values"](self.X, self.Y)
        self.assertEqual(mf, 898)
        self.assertIsInstance(mf, float)

    def test_percentage_of_Instances_with_missing_values(self):
        mf = self.mf["percentage_of_Instances_with_missing_values"](self.X, self.Y)
        self.assertAlmostEqual(mf, 1.0)
        self.assertIsInstance(mf, float)

    def test_number_of_features_with_missing_values(self):
        mf = self.mf["number_of_features_with_missing_values"](self.X, self.Y)
        self.assertEqual(mf, 29)
        self.assertIsInstance(mf, float)

    def test_percentage_of_features_with_missing_values(self):
        mf = self.mf["percentage_of_features_with_missing_values"](self.X, self.Y)
        self.assertAlmostEqual(mf, float(29)/float(38))
        self.assertIsInstance(mf, float)

    def test_number_of_missing_values(self):
        mf = self.mf["number_of_missing_values"](self.X, self.Y)
        self.assertEqual(mf, 22175)
        self.assertIsInstance(mf, float)

    def test_percentage_missing_values(self):
        mf = self.mf["percentage_of_missing_values"](self.X, self.Y)
        self.assertAlmostEqual(mf, float(22175)/float((38*898)))
        self.assertIsInstance(mf, float)

    def test_number_of_numeric_features(self):
        mf = self.mf["number_of_numeric_features"](self.X, self.Y)
        self.assertEqual(mf, 6)
        self.assertIsInstance(mf, float)

    def test_number_of_categorical_features(self):
        mf = self.mf["number_of_categorical_features"](self.X, self.Y)
        self.assertEqual(mf, 32)
        self.assertIsInstance(mf, float)

    def test_ratio_numerical_to_categorical(self):
        mf = self.mf["ratio_numerical_to_categorical"](self.X, self.Y)
        self.assertAlmostEqual(mf, float(6)/float(32))
        self.assertIsInstance(mf, float)

    def test_ratio_categorical_to_numerical(self):
        mf = self.mf["ratio_categorical_to_numerical"](self.X, self.Y)
        self.assertAlmostEqual(mf, float(32)/float(6))
        self.assertIsInstance(mf, float)

    def test_dataset_ratio(self):
        mf = self.mf["dataset_ratio"](self.X, self.Y)
        self.assertAlmostEqual(mf, float(38)/float(898))
        self.assertIsInstance(mf, float)

    def test_inverse_dataset_ratio(self):
        mf = self.mf["inverse_dataset_ratio"](self.X, self.Y)
        self.assertAlmostEqual(mf, float(898)/float(38))
        self.assertIsInstance(mf, float)

    def test_class_probability_min(self):
        mf = self.mf["class_probability_min"](self.X, self.Y)
        self.assertAlmostEqual(mf, float(8)/float(898))
        self.assertIsInstance(mf, float)

    def test_class_probability_max(self):
        mf = self.mf["class_probability_max"](self.X, self.Y)
        self.assertAlmostEqual(mf, float(684)/float(898))
        self.assertIsInstance(mf, float)

    def test_class_probability_mean(self):
        mf = self.mf["class_probability_mean"](self.X, self.Y)
        classes = np.array((8, 99, 684, 67, 40), dtype=np.float64)
        prob_mean = (classes / float(898)).mean()
        self.assertAlmostEqual(mf, prob_mean)
        self.assertIsInstance(mf, float)

    def test_class_probability_std(self):
        mf = self.mf["class_probability_std"](self.X, self.Y)
        classes = np.array((8, 99, 684, 67, 40), dtype=np.float64)
        prob_std = (classes / float(898)).std()
        self.assertAlmostEqual(mf, prob_std)
        self.assertIsInstance(mf, float)

    def test_symbols_min(self):
        mf = self.mf["symbols_min"](self.X, self.Y)
        self.assertEqual(mf, 1)

    def test_symbols_max(self):
        # this is attribute steel
        mf = self.mf["symbols_max"](self.X, self.Y)
        self.assertEqual(mf, 7)

    def test_symbols_mean(self):
        mf = self.mf["symbols_mean"](self.X, self.Y)
        # Empty looking spaces denote empty attributes
        symbol_frequency = [2, 1, 7, 1, 2, 4, 1, 1, 4, 2, 1, 1, 1, 2, 1, #
                            1, 1, 1,   1, 1,    3, 1,           2, 2, 3, 2]
        self.assertAlmostEqual(mf, np.mean(symbol_frequency))

    def test_symbols_std(self):
        mf = self.mf["symbols_std"](self.X, self.Y)
        symbol_frequency = [2, 1, 7, 1, 2, 4, 1, 1, 4, 2, 1, 1, 1, 2, 1, #
                            1, 1, 1,   1, 1,    3, 1,           2, 2, 3, 2]
        self.assertAlmostEqual(mf, np.std(symbol_frequency))

    def test_symbols_sum(self):
        mf = self.mf["symbols_sum"](self.X, self.Y)
        self.assertEqual(mf, 49)

    def test_kurtosis_min(self):
        # TODO: somehow compute the expected output?
        mf = self.mf["kurtosis_min"](self.X, self.Y)

    def test_kurtosis_max(self):
        # TODO: somehow compute the expected output?
        mf = self.mf["kurtosis_max"](self.X, self.Y)

    def test_kurtosis_mean(self):
        # TODO: somehow compute the expected output?
        mf = self.mf["kurtosis_mean"](self.X, self.Y)

    def test_kurtosis_std(self):
        # TODO: somehow compute the expected output?
        mf = self.mf["kurtosis_std"](self.X, self.Y)

    def test_skewness_min(self):
        # TODO: somehow compute the expected output?
        mf = self.mf["skewness_min"](self.X, self.Y)

    def test_skewness_max(self):
        # TODO: somehow compute the expected output?
        mf = self.mf["skewness_max"](self.X, self.Y)

    def test_skewness_mean(self):
        # TODO: somehow compute the expected output?
        mf = self.mf["skewness_mean"](self.X, self.Y)

    def test_skewness_std(self):
        # TODO: somehow compute the expected output?
        mf = self.mf["skewness_std"](self.X, self.Y)

    def test_class_entropy(self):
        mf = self.mf["class_entropy"](self.X, self.Y)
        classes = np.array((8, 99, 684, 67, 40), dtype=np.float64)
        classes = classes / sum(classes)
        entropy = -np.sum([c * np.log2(c) for c in classes])
        self.assertAlmostEqual(mf, entropy)

    def test_landmark_lda(self):
        # TODO: somehow compute the expected output?
        mf = self.mf["landmark_lda"](self.Xnp, self.Ynp)

    def test_landmark_naive_bayes(self):
        # TODO: somehow compute the expected output?
        mf = self.mf["landmark_naive_bayes"](self.Xnp, self.Ynp)

    def test_landmark_decision_tree(self):
        # TODO: somehow compute the expected output?
        mf = self.mf["landmark_decision_tree"](self.Xnp, self.Ynp)

    def test_decision_node(self):
        # TODO: somehow compute the expected output?
        mf = self.mf["landmark_decision_node_learner"](self.Xnp, self.Ynp)

    def test_random_node(self):
        # TODO: somehow compute the expected output?
        mf = self.mf["landmark_random_node_learner"](self.Xnp, self.Ynp)

    @unittest.skip("Currently not implemented!")
    def test_worst_node(self):
        # TODO: somehow compute the expected output?
        mf = self.mf["landmark_worst_node_learner"](self.Xnp, self.Ynp)

    def test_1NN(self):
        # TODO: somehow compute the expected output?
        mf = self.mf["landmark_1NN"](self.Xnp, self.Ynp)

    def test_pca(self):
        hf = self.helpers["PCA"](self.Xnp, self.Ynp)

    def test_pca_95percent(self):
        # TODO: somehow compute the expected output?
        mf = self.mf["pca_95percent"](self.Xnp, self.Ynp)
        self.assertNotAlmostEqual(1, mf)

    def test_pca_kurtosis_first_pc(self):
        # TODO: somehow compute the expected output?
        mf = self.mf["pca_kurtosis_first_pc"](self.Xnp, self.Ynp)
        self.assertNotAlmostEqual(0, mf)

    def test_pca_skewness_first_pc(self):
        # TODO: somehow compute the expected output?
        mf = self.mf["pca_skewness_first_pc"](self.Xnp, self.Ynp)
        self.assertNotAlmostEqual(0, mf)

    def test_calculate_all_metafeatures(self):
        mf = meta_features.calculate_all_metafeatures(self.ds)


if __name__ == "__main__":
    #suite = unittest.TestLoader().loadTestsFromTestCase(TestMetaFeatures)
    #unittest.TextTestRunner(verbosity=2).run(suite)
    t = unittest.TestLoader().loadTestsFromName(
        "pyMetaLearn.metafeatures.test_meta_features.TestMetaFeatures"
        ".test_calculate_all_metafeatures")
    unittest.TextTestRunner(verbosity=2).run(t)