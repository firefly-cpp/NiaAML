NiaAML's documentation!
==================================

.. automodule:: niaaml

NiaAML is an automated machine learning Python framework based on nature-inspired algorithms for optimization. The name comes from the automated machine learning method of the same name [1]. Its goal is to efficiently compose the best possible classification pipeline for the given task using components on the input. The components are divided into three groups: feature seletion algorithms, feature transformation algorithms and classifiers. The framework uses nature-inspired algorithms for optimization to choose the best set of components for the classification pipeline on the output and optimize their parameters. We use `NiaPy framework <https://github.com/NiaOrg/NiaPy>`_  for the optimization process which is a popular Python collection of nature-inspired algorithms. The NiaAML framework is easy to use and customize or expand to suit your needs.

* **Free software:** MIT license
* **Github repository:** https://github.com/lukapecnik/NiaAML
* **Python versions:** 3.6.x, 3.7.x, 3.8.x

The main documentation is organized into a couple of sections:

* :ref:`user-docs`
* :ref:`dev-docs`
* :ref:`about-docs`

.. _user-docs:

.. toctree::
   :maxdepth: 3
   :caption: User Documentation

   getting_started

.. _dev-docs:

.. toctree::
   :maxdepth: 2
   :caption: Developer Documentation

   changelog
   installation
   testing
   documentation
   api/index

.. _about-docs:

.. toctree::
   :maxdepth: 3
   :caption: About

   about
   contributing
   code_of_conduct

References
----------

[1] Iztok Fister Jr., Milan Zorman, Dušan Fister, Iztok Fister. Continuous optimizers for automatic design and evaluation of classification pipelines. In: Frontier applications of nature inspired computation. Springer tracts in nature-inspired computing, pp.281-301, 2020.