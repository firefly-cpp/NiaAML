Getting Started
===============

This section is going to show you how to use the NiaAML framework. First install NiaAML package using the following command:

.. code:: bash

    pip3 install niaaml

After the successful installation you are ready to run your first example.

Basic example
-------------
Create a new file, with name, for example *my_first_pipeline.py* and paste in the code below.

.. code:: python

    from niaaml import PipelineOptimizer, Pipeline
    from niaaml.data import BasicDataReader
    import numpy

    # dummy random data
    data_reader = BasicDataReader(
        x=numpy.random.uniform(low=0.0, high=15.0, size=(50, 3)),
        y=numpy.random.choice(['Class 1', 'Class 2'], size=50)
    )

    pipeline_optimizer = PipelineOptimizer(
        data=data_reader,
        classifiers=['AdaBoost', 'Bagging', 'MultiLayerPerceptron', 'RandomForest', 'ExtremelyRandomizedTrees', 'LinearSVC'],
        feature_selection_algorithms=['SelectKBest', 'SelectPercentile', 'ParticleSwarmOptimization', 'VarianceThreshold'],
        feature_transform_algorithms=['Normalizer', 'StandardScaler']
    )
    pipeline = pipeline_optimizer.run('Accuracy', 15, 15, 300, 300, 'ParticleSwarmAlgorithm', 'ParticleSwarmAlgorithm')

**As you can see, pipeline components, fitness function and optimization algorithms are always passed into pipeline optimization using their class names.** The example below uses the Particle Swarm Algorithm as the optimization algorithm. You can find a list of all available algorithms in the `NiaPy's documentation <https://niapy.readthedocs.io/en/stable/>`_.
Now you can run it using the command ``python3 my_first_pipeline.py``. The code currently does not do much, but we can save our pipeline to a file so we can use it later or save a user-friendly representation of it to a text file. You can choose one or both of the scenarios by adding the code below.

.. code:: python

    pipeline.export('pipeline.ppln')
    pipeline.export_text('pipeline.txt')

If you want to load and use the saved pipeline later, you can use the following code.

.. code:: python
    
    from niaaml import Pipeline
    import pandas

    loaded_pipeline = Pipeline.load('pipeline.ppln')

    # some features (can be loaded using DataReader object instances)
    x = pandas.DataFrame([[0.35, 0.46, 5.32], [0.16, 0.55, 12.5]])
    y = loaded_pipeline.run(x)

**The framework also supports the original version of optimization process where the components selection and hyperparameter optimization steps are combined into one. You can replace the ``run`` method with the following code.**

.. code:: python
    
    pipeline = pipeline_optimizer.run_v1('Accuracy', 15, 400, 'ParticleSwarmAlgorithm')

This is a very simple example with dummy data. It is only intended to give you a basic idea on how to use the framework. **NiaAML supports numerical and categorical features.**

Find more examples `here <https://github.com/lukapecnik/NiaAML/tree/master/examples>`_

Components
----------

In the following sections you can see a list of currently implemented components divided into groups: classifiers, feature selection algorithms and feature transformation algorithms. At the end you can also see a list of currently implemented fitness functions for the optimization process. Values in parentheses are associated names.

Classifiers
^^^^^^^^^^^

* Adaptive Boosting (AdaBoost),
* Bagging (Bagging),
* Extremely Randomized Trees (ExtremelyRandomizedTrees),
* Linear SVC (LinearSVC),
* Multi Layer Perceptron (MultiLayerPerceptron),
* Random Forest Classifier (RandomForest),
* Decision Tree Classifier (DecisionTree),
* K-Neighbors Classifier (KNeighbors),
* Gaussian Process Classifier (GaussianProcess),
* Gaussian Naive Bayes (GaussianNB),
* Quadratic Discriminant Analysis (QuadraticDiscriminantAnalysis).

Feature Selection Algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Select K Best (SelectKBest),
* Select Percentile (SelectPercentile),
* Variance Threshold (VarianceThreshold).

Nature-Inspired
"""""""""""""""

* Bat Algorithm (BatAlgorithm),
* Differential Evolution (DifferentialEvolution),
* Self-Adaptive Differential Evolution (jDEFSTH),
* Grey Wolf Optimizer (GreyWolfOptimizer),
* Particle Swarm Optimization (ParticleSwarmOptimization).

Feature Transformation Algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Normalizer (Normalizer),
* Standard Scaler (StandardScaler),
* Maximum Absolute Scaler (MaxAbsScaler),
* Quantile Transformer (QuantileTransformer),
* Robust Scaler (RobustScaler).

Fitness Functions based on
^^^^^^^^^^^^^^^^^^^^^^^^^^

* Accuracy (Accuracy),
* Cohen's kappa (CohenKappa),
* F1-Score (F1),
* Precision (Precision).

Categorical Feature Encoders
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* One-Hot Encoder (OneHotEncoder).

Feature Imputers
^^^^^^^^^^^^^^^^

* Simple Imputer (SimpleImputer).

Optimization Algorithms
^^^^^^^^^^^^^^^^^^^^^^^

For the list of available optimization algorithms please see the `NiaPy's documentation <https://niapy.readthedocs.io/en/stable/>`_.

Optimization Process And Parameter Tuning
-----------------------------------------

In NiaAML there are two types of optimization. Goal of the first type is to find an optimal set of components (feature selection algorithm, feature transformation algorithm and classifier). The next step is to find optimal parameters for the selected set of components and that is a goal of the second type of optimization. Each component has an attribute **_params**, which is a dictionary of parameters and their possible values.

.. code:: python

    self._params = dict(
        n_estimators = ParameterDefinition(MinMax(min=10, max=111), np.uint),
        algorithm = ParameterDefinition(['SAMME', 'SAMME.R'])
    )

An individual in the second type of optimization is a real-valued vector that has a size equal to the sum of number of keys in all three dictionaries (classifier's _params, feature transformation algorithm's _params and feature selection algorithm's _params) and a value of each dimension is in range [0.0, 1.0]. The second type of optimization maps real values from the individual's vector to those parameter definitions in the dictionaries. Each parameter's value can be defined as a range or array of values. In the first case, a value from vector is mapped from one iterval to another and in the second case, a value from vector falls into one of the bins that represent an index of the array that holds possible parameter's values.

Let's say we have a classifier with 3 parameters, feature selection algorithm with 2 parameters and feature transformation algorithm with 4 parameters. Size of an individual in the second type of optimization is 9. Size of an individual in the first type of optimization is always 3 (1 classifier, 1 feature selection algorithm and 1 feature transform algorithm).

In some cases we may want to tune a parameter that needs additional information for setting its range of values, so we cannot set the range in the initialization method. In that case we should set its value in the dictionary to None and define it later in the process. The parameter will be a part of parameter tuning process as soon as we define its possible values. For example, see the implementation of :class:`niaaml.preprocessing.feature_selection.SelectKBest` and its parameter **k**.