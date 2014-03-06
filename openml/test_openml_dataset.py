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

    def test_convert_arff_structure_to_npy_real_targets(self):
        self.arff_object['attributes'][-1] = "REAL"
        values = itertools.cycle([0, 1, 2, 3, 4, 5])
        for instance in self.arff_object['data']:
            instance[-1] = values.next()
        array = self.ds.convert_arff_structure_to_npy(self.arff_object)
        self.assertEqual(array.shape, (898, 56))
        self.assertEqual(array[:-1].min(), 0)
        self.assertEqual(array[:-1].max(), 5)

    def test_convert_arff_structure_to_pandas(self):
        self.ds.convert_arff_structure_to_pandas(self.arff_object)

