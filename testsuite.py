__author__ = 'feurerm'


import unittest

import file_handling
import file_handling.testsuite
import openml
import openml.testsuite
import rundata
import rundata.testsuite
import skdata_
import skdata_.testsuite
import uci
import uci.testsuite


def suite():
    _suite = unittest.TestSuite()
    _suite.addTest(file_handling.testsuite.suite())
    _suite.addTest(openml.testsuite.suite())
    _suite.addTest(rundata.testsuite.suite())
    _suite.addTest(skdata_.testsuite.suite())
    _suite.addTest(uci.testsuite.suite())
    return _suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    test_suite = suite()
    runner.run(suite())

