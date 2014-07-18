import setuptools

# TODO: add the constraints module because of the arrow alignment in the plots
setuptools.setup(name="pyMetaLearn",
                 description="Metalearning utilities for python.",
                 version="0.1dev",
                 packages=setuptools.find_packages(),
                 package_data={'': ['*.txt', '*.md']},
                 author="Matthias Feurer",
                 author_email="feurerm@informatik.uni-freiburg.de",
                 license="BSD",
                 platforms=['Linux'],
                 classifiers=[],
                 url="github.com/mfeurer/pymetalearn")