# NiaAML

NiaAML is an automated machine learning Python framework based on nature-inspired algorithms for optimization. The name comes from the automated machine learning method of the same name [[1]](#1). Its goal is to efficiently compose the best possible classification pipeline for the given task using components on the input. The components are divided into three groups: feature seletion algorithms, feature transformation algorithms and classifiers. The framework uses nature-inspired algorithms for optimization to choose the best set of components for the classification pipeline on the output and optimize their parameters. We use <a href="https://github.com/NiaOrg/NiaPy">NiaPy framework</a> for the optimization process which is a popular Python collection of nature-inspired algorithms. The NiaAML framework is easy to use and customize or expand to suit your needs.

## Components

Below you can see a list of currently implemented components divided into three groups: classifiers, feature selection algorithms and feature transformation algorithms.

### Classifiers

* AdaBoost,
* Bagging,
* Extremely Randomized Trees,
* Linear SVC,
* Multi Layer Perceptron,
* Random Forest Classifier.

### Feature Selection Algorithms

* Select K Best,
* Select Percentile,
* Variance Threshold.

#### Nature-Inspired

* Bat Algorithm,
* Differential Evolution,
* Self-Adaptive Differential Evolution (jDEFSTH),
* Grey Wolf Optimizer,
* Particle Swarm Optimization.

### Feature Transformation Algorithms

* Normalizer,
* Standard Scaler.

## Examples

### Example of Usage

Load data and try to find the optimal pipeline for the given components.

```sh
from niaaml import PipelineOptimizer
from niaaml.data import CSVDataReader

data_reader = CSVDataReader(src='path_to_csv_file.csv', contains_classes = True, has_header = False)

pipeline_optimizer = PipelineOptimizer(
    data=data_reader,
    classifiers=['AdaBoost', 'Bagging', 'MultiLayerPerceptron', 'RandomForestClassifier'],
    feature_selection_algorithms=['SelectKBest', 'SelectPercentile', 'ParticleSwarmOptimization'],
    feature_transform_algorithms=['Normalizer', 'StandardScaler']
)
final_pipeline = t.run('Accuracy', 20, 20, 400, 400, 'ParticleSwarmAlgorithm', 'ParticleSwarmAlgorithm')
```

You can save a result of the optimization process as an object to a file for later use.

```sh
final_pipeline.export('pipeline.ppln')
```

And also load it from a file and use the pipeline.

```sh
loaded_pipeline = Pipeline.load('pipeline.ppln')

import numpy
# numpy array containing features
x = numpy.ndarray([[0.35, 0.46, 5.32], [0.16, 0.55, 12.5]], dtype=float)

y = loaded_pipeline.run(x)
```

You can also save a user-friendly representation of a pipeline to a text file.

```sh
final_pipeline.export_text('pipeline.txt')
```

### Example of a Pipeline Component Implementation

NiaAML framework is easily expandable as you can implement components by overriding the base classes' methods. To implement a classifier you should inherit from the [Classifier](niaaml/classifiers/classifier.py) class and you can do the same with [FeatureSelectionAlgorithm](niaaml/preprocessing/feature_selection/feature_selection_algorithm.py) and [FeatureTransformAlgorithm](niaaml/preprocessing/feature_transform/feature_transform_algorithm.py) classes. All of the mentioned classes inherit from the PipelineComponent class.

For more information take a look at the [Classifier](niaaml/classifiers/classifier.py) class and the implementation of the [AdaBoost](niaaml/classifiers/ada_boost.py) classifier that inherits from it.

## Licence

This package is distributed under the MIT License. This license can be found online at <http://www.opensource.org/licenses/MIT>.

## Disclaimer

This framework is provided as-is, and there are no guarantees that it fits your purposes or that it is bug-free. Use it at your own risk!

## References

<a id="1">[1]</a> Iztok Fister Jr., Milan Zorman, Dušan Fister, Iztok Fister. <a href="https://link.springer.com/chapter/10.1007%2F978-981-15-2133-1_13">Continuous optimizers for automatic design and evaluation of classification pipelines</a>. In: Frontier applications of nature inspired computation. Springer tracts in nature-inspired computing, pp.281-301, 2020.