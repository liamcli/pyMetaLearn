from unittest import TestCase
import os

import numpy as np
from openml.openml_dataset import OpenMLDataset
import metafeatures.metafeatures as meta_features
import arff

__author__ = 'feurerm'


class TestMetaFeatures(TestCase):
    def setUp(self):
        os.chdir(os.path.dirname(__file__))
        fh = open("../openml/test_cache/did1_anneal.arff")
        self.arff_object = arff.load(fh)
        fh.close()
        self.ds = OpenMLDataset("OpenML", 1, "anneal", None, None, "arff",
                                None, None, None, False)
        data_frame = self.ds.convert_arff_structure_to_pandas(self.arff_object)
        class_ = data_frame.keys()[-1]
        attributes = data_frame.keys()[0:-1]
        self.X = data_frame[attributes]
        self.Y = data_frame[class_]
        self.mf = meta_features.metafeatures

    def test_number_of_instance(self):
        self.assertEqual(self.mf["number_of_instances"](self.X, self.Y), 898)

    def test_number_of_classes(self):
        self.assertEqual(self.mf["number_of_classes"](self.X, self.Y), 5)

    def test_number_of_features(self):
        self.assertEqual(self.mf["number_of_features"](self.X, self.Y), 38)

    def test_number_of_Instances_with_missing_values(self):
        self.assertEqual(self.mf["number_of_Instances_with_missing_values"](
            self.X, self.Y), 898)

    def test_percentage_of_Instances_with_missing_values(self):
        self.assertAlmostEqual(self.mf["percentage_of_Instances_with_missing_values"](
            self.X, self.Y), 1.0)

    def test_number_of_features_with_missing_values(self):
        self.assertEqual(self.mf["number_of_features_with_missing_values"](
            self.X, self.Y), 29)

    def test_percentage_of_features_with_missing_values(self):
        self.assertAlmostEqual(self.mf["percentage_of_features_with_missing_values"](
            self.X, self.Y), float(29)/float(38))

    def test_number_of_missing_values(self):
        self.assertEqual(self.mf["number_of_missing_values"](
            self.X, self.Y), 22175)

    def test_percentage_missing_values(self):
        self.assertAlmostEqual(self.mf["percentage_of_missing_values"](
            self.X, self.Y), float(22175)/float((38*898)))

    def test_number_of_numeric_features(self):
        self.assertEqual(self.mf["number_of_numeric_features"](self.X,
                                                               self.Y), 6)

    def test_number_of_categorical_features(self):
        self.assertEqual(self.mf["number_of_categorical_features"](self.X,
                                                               self.Y), 32)

    def test_ratio_numerical_to_categorical(self):
        self.assertAlmostEqual(self.mf["ratio_numerical_to_categorical"]
            (self.X, self.Y), float(6)/float(32))

    def test_ratio_categorical_to_numerical(self):
        self.assertAlmostEqual(self.mf["ratio_categorical_to_numerical"]
            (self.X, self.Y), float(32)/float(6))

    def test_dimensionality(self):
        self.assertAlmostEqual(self.mf["dimensionality"](self.X, self.Y),
            float(38)/float(898))

    def test_class_probability_min(self):
        self.assertAlmostEqual(self.mf["class_probability_min"](self.X, self.Y),
            float(8)/float(898))

    def test_class_probability_max(self):
        self.assertAlmostEqual(self.mf["class_probability_max"](self.X, self.Y),
            float(684)/float(898))

    def test_class_probability_mean(self):
        classes = np.array((8, 99, 684, 67, 40), dtype=np.float64)
        prob_mean = (classes / float(898)).mean()
        self.assertAlmostEqual(self.mf["class_probability_mean"](self.X,
                                                                self.Y),
            prob_mean)

    def test_class_probability_std(self):
        classes = np.array((8, 99, 684, 67, 40), dtype=np.float64)
        prob_std = (classes / float(898)).std()
        self.assertAlmostEqual(self.mf["class_probability_std"](self.X,
                                                                self.Y),
            prob_std)