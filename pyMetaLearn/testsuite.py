__author__ = 'feurerm'


import unittest


from pyMetaLearn.metafeatures.test_meta_features import TestMetaFeatures
from pyMetaLearn.metalearning.test_meta_base import MetaBaseTest
import pyMetaLearn.openml.testsuite
from pyMetaLearn.optimizers.gridsearch.test_gridsearch import GridSearchTest
from pyMetaLearn.optimizers.metalearn_optimizer.test_metalearner import MetaLearnerTest
# SMAC warmstart
# Spearmint warmstart
import pyMetaLearn.skdata_.testsuite
import pyMetaLearn.smac_utils.testsuite
import pyMetaLearn.workflows.test_workflow_openml_to_libsvm


def suite():
    _suite = unittest.TestSuite()
    _suite.addTest(unittest.makeSuite(TestMetaFeatures))
    _suite.addTest(unittest.makeSuite(MetaBaseTest))
    _suite.addTest(pyMetaLearn.openml.testsuite.suite())
    _suite.addTest(unittest.makeSuite(GridSearchTest))
    _suite.addTest(unittest.makeSuite(MetaLearnerTest))
    _suite.addTest(pyMetaLearn.skdata_.testsuite.suite())

    return _suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    test_suite = suite()
    runner.run(suite())

