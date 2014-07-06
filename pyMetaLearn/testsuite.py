__author__ = 'feurerm'


import unittest

import pyMetaLearn.file_handling
import pyMetaLearn.file_handling.testsuite
import pyMetaLearn.metafeatures.test_meta_features
import pyMetaLearn.openml
import pyMetaLearn.openml.testsuite
import pyMetaLearn.rundata
import pyMetaLearn.rundata.testsuite
import pyMetaLearn.skdata_
import pyMetaLearn.skdata_.testsuite
#import uci
#import uci.testsuite


def suite():
    _suite = unittest.TestSuite()
    _suite.addTest(pyMetaLearn.file_handling.testsuite.suite())
    _suite.addTest(unittest.makeSuite(pyMetaLearn.metafeatures
                                      .test_meta_features.TestMetaFeatures))
    _suite.addTest(pyMetaLearn.openml.testsuite.suite())
    _suite.addTest(pyMetaLearn.rundata.testsuite.suite())
    _suite.addTest(pyMetaLearn.skdata_.testsuite.suite())
    #_suite.addTest(uci.testsuite.suite())
    return _suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    test_suite = suite()
    runner.run(suite())

