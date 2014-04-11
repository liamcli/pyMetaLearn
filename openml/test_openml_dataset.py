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
        fh = open("test_cache/did1_anneal.arff")
        self.arff_object = arff.load(fh)
        fh.close()
        self.ds = OpenMLDataset("OpenML", 1, "anneal", None, None, "arff",
                               None, None, None, False)

    def test_convert_arff_structure_to_npy_targets(self):
        self.arff_object['attributes'][-1] = "REAL"
        values = itertools.cycle([0, 1, 2, 3, 4, 5])
        for instance in self.arff_object['data']:
            instance[-1] = values.next()
        X, Y = self.ds._convert_arff_structure_to_npy(self.arff_object)
        self.assertEqual(X.shape, (898, 55))
        self.assertEqual(X.min(), 0)
        self.assertEqual(X.max(), 4880)

        X, Y = self.ds._convert_arff_structure_to_npy(self.arff_object,
                                                      scaling="normalize")
        self.assertEqual(X.min(), 0)
        self.assertEqual(X.max(), 1)

        self.assertEqual(Y.shape, (898, 1))

    def test_convert_arff_structure_to_pandas(self):
        self.ds.get_pandas(self.arff_object)

