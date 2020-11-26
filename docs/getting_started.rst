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