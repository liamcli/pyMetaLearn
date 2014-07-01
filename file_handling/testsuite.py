__author__ = 'feurerm'


import unittest
import test_csv_handler


def suite():
    _suite = unittest.TestSuite()
    _suite.addTest(unittest.makeSuite(test_csv_handler.TestCSVHandler))
    return _suite


if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite())