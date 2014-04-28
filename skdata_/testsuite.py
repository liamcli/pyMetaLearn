__author__ = 'feurerm'


import unittest


def suite():
    _suite = unittest.TestSuite()
    return _suite


if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite())
