Getting Started
===============

This section is going to show you how to use the NiaAML framework. First install NiaAML package using the following command:

.. code:: bash

    pip install niaaml

After the successful installation you are ready to run your first example.

Basic example
-------------
Create a new file, with name, for example *my_first_pipeline.py* and paste in the code below.

.. code:: python

    from niaaml import PipelineOptimizer
    from niaaml.data import BasicDataReader
    import numpy

    data_reader = BasicDataReader(
        numpy.ndarray([[1.23, 3.32, 43.4], [2.23, 3.33, 33.3]]),
        ['A', 'B']
    )

    pipeline_optimizer = PipelineOptimizer(
        data=data_reader,
        classifiers=['AdaBoost', 'Bagging', 'MultiLayerPerceptron', 'RandomForest', 'ExtremelyRandomizedTrees', 'LinearSVC'],
        feature_selection_algorithms=['SelectKBest', 'SelectPercentile', 'ParticleSwarmOptimization', 'VarianceThreshold'],
        feature_transform_algorithms=['Normalizer', 'StandardScaler']
    )
    final_pipeline = t.run('Accuracy', 20, 20, 400, 400, 'ParticleSwarmAlgorithm', 'ParticleSwarmAlgorithm')

**As you can see, pipeline components, fitness function and optimization algorithms are always passed into pipeline optimization using their class names.** The example below uses the Particle Swarm Algorithm as the optimization algorithm. You can find a list of all available algorithms in the `NiaPy's documentation <https://niapy.readthedocs.io/en/stable/>`_.
Now you can run it using the command ``python my_first_pipeline.py``. The code currently does not do much as we want to save our pipeline to a file so we can use it later or at least save a user-friendly representation of it to a text file. You can choose one or both scenarios by adding the code below.

.. code:: python

    final_pipeline.export('pipeline.ppln')
    final_pipeline.export_text('pipeline.txt')

If you want to load and use the saved pipeline later, you can use the following code.

.. code:: python
    
    from niaaml import Pipeline
    import numpy

    loaded_pipeline = Pipeline.load('pipeline.ppln')

    # some features (can be loaded using DataReader object instances)
    x = numpy.ndarray([[0.35, 0.46, 5.32], [0.16, 0.55, 12.5]], dtype=float)
    y = loaded_pipeline.run(x)

This is a very simple example with dummy data. It is only intended to give you a basic idea on how to use the framework. **NiaAML currently supports only numeric features. However, we are planning to add support for categorical features too.**

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
* Random Forest Classifier (RandomForestClassifier).

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
* Standard Scaler (StandardScaler).

Fitness Functions
^^^^^^^^^^^^^^^^^

* Accuracy (Accuracy),
* Cohen's kappa (CohenKappa),
* F1-Score (F1),
* Precision (Precision).

Optimization Algorithms
^^^^^^^^^^^^^^^^^^^^^^^

For the list of available optimization algorithms please see the `NiaPy's documentation <https://niapy.readthedocs.io/en/stable/>`_.