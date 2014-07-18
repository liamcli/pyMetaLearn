import unittest

import numpy as np

from pyMetaLearn.openml.openml_task import OpenMLTask


class TestOpenMLTask(unittest.TestCase):
    def test_get_fold(self):
        task = OpenMLTask(1, "supervised classification", 1, "class",
                          "crossvalidation wth holdout", None, None, None,
                          None, None)
        X = np.arange(20)
        Y = np.array(([0] * 10) + ([1] * 10))
        splits = task._get_fold(X, Y, 0, 2, shuffle=False)
        self.assertTrue(all(splits[0] ==
                            np.array([2, 4, 6, 8, 9, 10, 12, 14, 16, 19])))
        self.assertTrue(all(splits[1] ==
                             np.array([0, 1, 3, 5, 7, 11, 13, 15, 17, 18])))

        splits = task._get_fold(X, Y, 0, 2, shuffle=True)
        print splits
        self.assertTrue(all(splits[0] ==
                            np.array([1, 5, 3, 16, 9, 19, 12, 7, 10, 14])))
        self.assertTrue(all(splits[1] ==
                            np.array([ 0, 17, 15, 8, 11, 18, 13, 2, 4, 6])))
