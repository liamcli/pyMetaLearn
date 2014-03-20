from unittest import TestCase
import os

import numpy as np
from openml.openml_dataset import OpenMLDataset
import pyMetaLearn.metafeatures.metafeatures as meta_features
import arff

__author__ = 'feurerm'


class TestMetaFeatures(TestCase):
    def setUp(self):
        os.chdir(os.path.dirname(__file__))
        fh = open("../openml/test_cache/did1_anneal.arff")
        self.arff_object = arff.load(fh)
        fh.close()
        self.ds = OpenMLDataset("OpenML", 1, "anneal", None, None, "arff",
                                None, None,
                                "/home/feurerm/thesis/Software/pyMetaLearn/openml/test_cache", True)
        X, Y = self.ds.get_processed_files()
        self.X = X
        self.Y = Y
        self.mf = meta_features.metafeatures

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

    def test_dimensionality(self):
        mf = self.mf["dimensionality"](self.X, self.Y)
        self.assertAlmostEqual(mf, float(38)/float(898))
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