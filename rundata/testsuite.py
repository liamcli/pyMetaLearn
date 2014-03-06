__author__ = 'feurerm'


import unittest
import test_smac_extracter


def suite():
    _suite = unittest.TestSuite()
    _suite.addTest(unittest.makeSuite(test_smac_extracter.TestSMACExtracter))
    return _suite


if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite())