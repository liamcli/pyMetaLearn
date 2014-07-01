from distutils.core import setup
from Cython.Build import cythonize

import os

import numpy
from numpy.distutils.misc_util import Configuration



def configuration(parent_package="", top_path=None):
    config = Configuration("tree", parent_package, top_path)
    libraries = []
    if os.name == 'posix':
        libraries.append('m')
    config.add_extension("_tree",
                         sources=["_tree.c"],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries,
                         extra_compile_args=["-O3"]) #["-g"])#

    config.add_subpackage("tests")

    return config

setup(**configuration().todict())

#if __name__ == "__main__":
#    from numpy.distutils.core import setup
#    setup(name = 'correlation_criterion_', ext_modules = cythonize(
# "correlation_criterion_.pyx"))#,
#        # **configuration().todict())


