__author__ = 'feurerm'


import unittest
import test_openml_dataset
import test_manage_openml_data


def suite():
    _suite = unittest.TestSuite()
    _suite.addTest(unittest.makeSuite(test_openml_dataset.TestOpenMLDataset))
    _suite.addTest(unittest.makeSuite(test_manage_openml_data.
        TestManageOpenMLData))
    return _suite


if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite())