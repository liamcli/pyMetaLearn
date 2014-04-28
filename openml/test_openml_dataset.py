from unittest import TestCase
import numpy as np
import itertools
import arff

__author__ = 'feurerm'

import unittest
import os

from openml_dataset import OpenMLDataset


class TestOpenMLDataset(unittest.TestCase):
    def setUp(self):
        os.chdir(os.path.dirname(__file__))
        fh = open("test_cache/datasets.arff")
        self.arff_object = arff.load(fh)
        fh.close()
        self.ds = OpenMLDataset("OpenML", 1, "anneal", None, None, "arff",
                                None, None, None, False)

    def test_convert_arff_structure_to_npy_targets(self):
        values = itertools.cycle([0, 1, 2, 3, 4, 5])
        for instance in self.arff_object['data']:
            instance[-1] = values.next()
        X, Y = self.ds._convert_arff_structure_to_npy(self.arff_object)
        self.assertEqual(X.shape, (898, 55))
        self.assertEqual(X.min(), 0)
        self.assertEqual(X.max(), 4880)

        X, Y = self.ds._convert_arff_structure_to_npy(self.arff_object,
                                                      scaling="scale")
        self.assertEqual(X.min(), 0)
        self.assertEqual(X.max(), 1)

        self.assertEqual(Y.shape, (898,))

    def test_convert_arff_structure_to_pandas(self):
        self.ds._convert_arff_structure_to_pandas(self.arff_object)

    def test_convert_attribute_type_nominal(self):
        attribute_type = [u'?', u'GB', u'GK', u'GS', u'TN', u'ZA', u'ZF', u'ZH', u'ZM', u'ZS']
        attribute_type = self.ds._convert_attribute_type(attribute_type)
        self.assertEqual(attribute_type, 'object')

    def test_convert_attribute_type_tuple(self):
        attribute_type = (u'?', u'GB', u'GK', u'GS', u'TN', u'ZA', u'ZF',
                          u'ZH', u'ZM', u'ZS')
        self.assertRaises(NotImplementedError,
            self.ds._convert_attribute_type, attribute_type)

    def test_convert_attribute_type_real(self):
        attribute_type = 'REAL'
        attribute_type = self.ds._convert_attribute_type(attribute_type)
        self.assertEqual(attribute_type, np.float64)

    def test_convert_attribute_type_integer(self):
        attribute_type = 'INTEGER'
        attribute_type = self.ds._convert_attribute_type(attribute_type)
        self.assertEqual(attribute_type, np.float64)

    def test_convert_attribute_type_string(self):
        attribute_type = 'BLA'
        self.assertRaises(NotImplementedError,
            self.ds._convert_attribute_type, attribute_type)