Installation
============

Setup development environment
-----------------------------

Requirements
~~~~~~~~~~~~

-  Poetry: https://python-poetry.org/docs/

After installing Poetry and cloning the project from GitHub, you should
run the following command from the root of the cloned project:

.. code:: sh

    $ poetry install

All of the project's dependencies should be installed and the project
ready for further development. **Note that Poetry creates a separate
virtual environment for your project.**

Development dependencies
~~~~~~~~~~~~~~~~~~~~~~~~

List of NiaAML's dependencies:

+----------------+--------------+------------+
| Package        | Version      | Platform   |
+================+==============+============+
| numpy          | ^1.19.1      | All        |
+----------------+--------------+------------+
| scikit-learn   | ^1.1.2       | All        |
+----------------+--------------+------------+
| niapy          | ^2.0.5       | All        |
+----------------+--------------+------------+
| pandas         | ^2.1.1       | All        |
+----------------+--------------+------------+

List of development dependencies:

+--------------------+-----------+------------+
| Package            | Version   | Platform   |
+====================+===========+============+
| sphinx             | ^3.3.1    | Any        |
+--------------------+-----------+------------+
| sphinx-rtd-theme   | ^0.5.0    | Any        |
+--------------------+-----------+------------+
| coveralls          | ^2.2.0    | Any        |
+--------------------+-----------+------------+
| autoflake          | ^1.4      | Any        |
+--------------------+-----------+------------+
| black              | ^21.5b1   | Any        |
+--------------------+-----------+------------+
| pre-commit         | ^2.12.1   | Any        |
+--------------------+-----------+------------+
| pytest             | ^7.4.2    | Any        |
+--------------------+-----------+------------+
| pytest-cov         | ^4.1.0    | Any        |
+--------------------+-----------+------------+