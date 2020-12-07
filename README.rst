NiaAML
======

.. image:: https://travis-ci.com/lukapecnik/NiaAML.svg?branch=master
    :target: https://travis-ci.com/lukapecnik/NiaAML

.. image:: https://coveralls.io/repos/github/lukapecnik/NiaAML/badge.svg?branch=travisCI_integration
    :target: https://coveralls.io/github/lukapecnik/NiaAML?branch=travisCI_integration

.. image:: https://img.shields.io/pypi/v/niaaml.svg
    :target: https://pypi.python.org/pypi/niaaml

.. image:: https://img.shields.io/pypi/pyversions/niaaml.svg
    :target: https://pypi.org/project/NiaPy/

.. image:: https://img.shields.io/github/license/lukapecnik/niaaml.svg
    :target: https://github.com/lukapecnik/niaaml/blob/master/LICENSE

NiaAML is an automated machine learning Python framework based on
nature-inspired algorithms for optimization. The name comes from the
automated machine learning method of the same name [1]. Its
goal is to efficiently compose the best possible classification pipeline
for the given task using components on the input. The components are
divided into three groups: feature seletion algorithms, feature
transformation algorithms and classifiers. The framework uses
nature-inspired algorithms for optimization to choose the best set of
components for the classification pipeline on the output and optimize
their parameters. We use `NiaPy framework <https://github.com/NiaOrg/NiaPy>`_ for the optimization process
which is a popular Python collection of nature-inspired algorithms. The
NiaAML framework is easy to use and customize or expand to suit your
needs.

The NiaAML framework allows you not only to run full pipeline optimization, but also separate implemented components such as classifiers, feature selection algorithms, etc. **It supports numerical and categorical features.**

- **Documentation:** https://niaaml.readthedocs.io/en/latest/

Installation
------------

Install NiaAML with pip:

.. code:: sh

    pip install niaaml

In case you would like to try out the latest pre-release version of the framework, install it using:

.. code:: sh

    pip install niaaml --pre

Usage
-----

See the project's `repository <https://github.com/lukapecnik/NiaAML>`_ for usage examples.

Components
----------

In the following sections you can see a list of currently implemented 
components divided into groups: classifiers, feature selection 
algorithms and feature transformation algorithms. At the end you can 
also see a list of currently implemented fitness functions for the optimization process 
and categorical features' encoders.

Classifiers
~~~~~~~~~~~

-  Adaptive Boosting (AdaBoost),
-  Bagging (Bagging),
-  Extremely Randomized Trees (ExtremelyRandomizedTrees),
-  Linear SVC (LinearSVC),
-  Multi Layer Perceptron (MultiLayerPerceptron),
-  Random Forest Classifier (RandomForestClassifier).

Feature Selection Algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Select K Best (SelectKBest),
-  Select Percentile (SelectPercentile),
-  Variance Threshold (VarianceThreshold).

Nature-Inspired
^^^^^^^^^^^^^^^

-  Bat Algorithm (BatAlgorithm),
-  Differential Evolution (DifferentialEvolution),
-  Self-Adaptive Differential Evolution (jDEFSTH),
-  Grey Wolf Optimizer (GreyWolfOptimizer),
-  Particle Swarm Optimization (ParticleSwarmOptimization).

Feature Transformation Algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Normalizer (Normalizer),
-  Standard Scaler (StandardScaler).

Fitness Functions
~~~~~~~~~~~~~~~~~

-  Accuracy (Accuracy),
-  Cohen's kappa (CohenKappa),
-  F1-Score (F1),
-  Precision (Precision).

Categorical Feature Encoders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- One-Hot Encoder (OneHotEncoder).

Licence
-------

This package is distributed under the MIT License. This license can be
found online at http://www.opensource.org/licenses/MIT.

Disclaimer
----------

This framework is provided as-is, and there are no guarantees that it
fits your purposes or that it is bug-free. Use it at your own risk!

References
----------

[1] Iztok Fister Jr., Milan Zorman, Dušan Fister, Iztok Fister.
Continuous optimizers for automatic design and evaluation of
classification pipelines. In: Frontier applications of nature inspired
computation. Springer tracts in nature-inspired computing, pp.281-301,
2020.
