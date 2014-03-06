import unittest
import test_uci_parser


def suite():
    _suite = unittest.TestSuite()
    _suite.addTest(unittest.makeSuite(test_uci_parser.Test_UCI_Parser))
    return _suite


if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite())