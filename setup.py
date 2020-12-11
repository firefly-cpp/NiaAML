#!/usr/bin/env python

"""Setup script for the package."""

import io
import os
import sys
import logging

import setuptools


PACKAGE_NAME = 'niaaml'
MINIMUM_PYTHON_VERSION = '3.6.1'


def check_python_version():
    """Exit when the Python version is too low."""
    if sys.version < MINIMUM_PYTHON_VERSION:
        sys.exit("Python {0}+ is required.".format(MINIMUM_PYTHON_VERSION))


def read_package_variable(key, filename='__init__.py'):
    """Read the value of a variable from the package without importing."""
    module_path = os.path.join(PACKAGE_NAME, filename)
    with open(module_path) as module:
        for line in module:
            parts = line.strip().split(' ', 2)
            if parts[:-1] == [key, '=']:
                return parts[-1].strip("'")
    logging.warning("'%s' not found in '%s'", key, module_path)
    return None


def build_description():
    """Build a description for the project from documentation files."""
    try:
        readme = io.open("README.rst", encoding="UTF-8").read()
    except IOError:
        return "<placeholder>"
    else:
        return readme


check_python_version()

PACKAGE_VERSION = read_package_variable('__version__')

setuptools.setup(
    name=PACKAGE_NAME,
    version=PACKAGE_VERSION,
    description="""
        Python automated machine learning framework.
        """,
    url='https://github.com/lukapecnik/NiaAML',
    author='lukapecnik',
    author_email='lukapecnik96@gmail.com',
    packages=setuptools.find_packages(),
    long_description=build_description(),
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development'
    ],
    tests_requires=[
        'sphinx >= 3.3.1'
        'sphinx-rtd-theme >= 0.5.0'
        'coveralls >= 2.2.0'
    ],
    install_requires=[
        'numpy >= 1.19.1',
        'scikit-learn >= 0.23.2',
        'NiaPy >= 2.0.0rc11',
        'pandas >= 1.1.4'
    ]
)
