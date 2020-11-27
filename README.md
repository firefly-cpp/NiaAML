# NiaAML

NiaAML is an automated machine learning Python framework based on nature-inspired algorithms for optimization. The name comes from the automated machine learning method of the same name [[1]](#1). Its goal is to efficiently compose the best possible classification pipeline for the given task using components on the input. The components are divided into three groups: feature seletion algorithms, feature transformation algorithms and classifiers. The framework uses nature-inspired algorithms for optimization to choose the best set of components for the classification pipeline on the output and optimize their parameters. We use <a href="https://github.com/NiaOrg/NiaPy">NiaPy framework</a> for the optimization process which is a popular Python collection of nature-inspired algorithms. The NiaAML framework is easy to use and customize or expand to suit your needs.

## Installation

Install NiaAML with pip:

```sh
pip install niaaml
```

## Components

In the following sections you can see a list of currently implemented components divided into groups: classifiers, feature selection algorithms and feature transformation algorithms. At the end you can also see a list of currently implemented fitness functions for the optimization process. All of the components are passed into the optimization process using their class names. Let's say we want to choose between Adaptive Boosting, Bagging and Multi Layer Perceptron classifiers, Select K Best and Select Percentile feature selection algorithms and Normalizer as the feature transformation algorithm (may not be selected during the optimization process).

```python
PipelineOptimizer(
    data=...,
    classifiers=['AdaBoost', 'Bagging', 'MultiLayerPerceptron'],
    feature_selection_algorithms=['SelectKBest', 'SelectPercentile'],
    feature_transform_algorithms=['Normalizer']
)
```

For a full example see the [Examples section](#examples).

### Classifiers

* Adaptive Boosting (AdaBoost),
* Bagging (Bagging),
* Extremely Randomized Trees (ExtremelyRandomizedTrees),
* Linear SVC (LinearSVC),
* Multi Layer Perceptron (MultiLayerPerceptron),
* Random Forest Classifier (RandomForestClassifier).

### Feature Selection Algorithms

* Select K Best (SelectKBest),
* Select Percentile (SelectPercentile),
* Variance Threshold (VarianceThreshold).

#### Nature-Inspired

* Bat Algorithm (BatAlgorithm),
* Differential Evolution (DifferentialEvolution),
* Self-Adaptive Differential Evolution (jDEFSTH),
* Grey Wolf Optimizer (GreyWolfOptimizer),
* Particle Swarm Optimization (ParticleSwarmOptimization).

### Feature Transformation Algorithms

* Normalizer (Normalizer),
* Standard Scaler (StandardScaler).

### Fitness Functions

* Accuracy (Accuracy),
* Cohen's kappa (CohenKappa),
* F1-Score (F1),
* Precision (Precision).

## Examples

NiaAML framework currently supports only numeric features on the input. **However, we are planning to add support for categorical features too.**

### Example of Usage

Load data and try to find the optimal pipeline for the given components. The example below uses the Particle Swarm Algorithm as the optimization algorithm. You can find a list of all available algorithms in the <a href="https://niapy.readthedocs.io/en/stable/">NiaPy's documentation</a>.

```python
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
```

You can save a result of the optimization process as an object to a file for later use.

```python
final_pipeline.export('pipeline.ppln')
```

And also load it from a file and use the pipeline.

```python
loaded_pipeline = Pipeline.load('pipeline.ppln')

import numpy
# some features (can be loaded using DataReader object instances)
x = numpy.ndarray([[0.35, 0.46, 5.32], [0.16, 0.55, 12.5]], dtype=float)

y = loaded_pipeline.run(x)
```

You can also save a user-friendly representation of a pipeline to a text file.

```python
final_pipeline.export_text('pipeline.txt')
```

This is a very simple example with dummy data. It is only intended to give you a basic idea on how to use the framework. **In practice we recomend you to load data using other implementations of the DataReader class (e.g. CSVDataReader).**

### Example of a Pipeline Component Implementation

NiaAML framework is easily expandable as you can implement components by overriding the base classes' methods. To implement a classifier you should inherit from the [Classifier](niaaml/classifiers/classifier.py) class and you can do the same with [FeatureSelectionAlgorithm](niaaml/preprocessing/feature_selection/feature_selection_algorithm.py) and [FeatureTransformAlgorithm](niaaml/preprocessing/feature_transform/feature_transform_algorithm.py) classes. All of the mentioned classes inherit from the [PipelineComponent](niaaml/pipeline_component.py) class.

For more information take a look at the [Classifier](niaaml/classifiers/classifier.py) class and the implementation of the [AdaBoost](niaaml/classifiers/ada_boost.py) classifier that inherits from it.

### Fitness Functions

NiaAML framework also allows you to implement your own fitness function. All you need to do is implement the [FitnessFunction](niaaml/fitness/fitness_function.py) class.

For more information take a look at the [Accuracy](niaaml/fitness/accuracy.py) implementation.

<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-13-orange.svg?style=flat-square)](#contributors)
<!-- ALL-CONTRIBUTORS-BADGE:END --> 

## Licence

This package is distributed under the MIT License. This license can be found online at <http://www.opensource.org/licenses/MIT>.

## Disclaimer

This framework is provided as-is, and there are no guarantees that it fits your purposes or that it is bug-free. Use it at your own risk!

## References

<a id="1">[1]</a> Iztok Fister Jr., Milan Zorman, Dušan Fister, Iztok Fister. <a href="https://link.springer.com/chapter/10.1007%2F978-981-15-2133-1_13">Continuous optimizers for automatic design and evaluation of classification pipelines</a>. In: Frontier applications of nature inspired computation. Springer tracts in nature-inspired computing, pp.281-301, 2020.