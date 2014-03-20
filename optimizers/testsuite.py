import unittest
import test_optimizer_base
import gridsearch.test_gridsearch as test_gridsearch
import metalearn_optimizer.test_meta_base as test_meta_base
import metalearn_optimizer.test_metalearner as test_metalearner


def suite():
    _suite = unittest.TestSuite()
    _suite.addTest(unittest.makeSuite(test_optimizer_base.OptimizerBaseTest))
    _suite.addTest(unittest.makeSuite(test_gridsearch.GridSearchTest))
    _suite.addTest(unittest.makeSuite(test_meta_base.MetaBaseTest))
    _suite.addTest(unittest.makeSuite(test_metalearner.MetaLearnerTest))
    return _suite


if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite())