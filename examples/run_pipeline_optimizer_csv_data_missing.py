import os
from niaaml import PipelineOptimizer
from niaaml.data import CSVDataReader

"""
In this example, we show how to use the PipelineOptimizer class. This example is using an instance of CSVDataReader.
The instantiated PipelineOptimizer will try and assemble the best pipeline with the components that are specified in its constructor.
We use a dataset with 1 categorical feature and missing values to demonstrate a use of PipelineOptimizer instance with automatic feature encoding and imputation.
"""

# prepare data reader using csv file
data_reader = CSVDataReader(
    src=os.path.dirname(os.path.abspath(__file__))
    + "/example_files/dataset_categorical_missing.csv",
    has_header=False,
    contains_classes=True,
)

# instantiate PipelineOptimizer that chooses among specified classifiers, feature selection algorithms and feature transform algorithms
# OneHotEncoder is used for encoding categorical features in this example
# SimpleImputer is used for imputing missing values in this example
# log is True by default, log_verbose means more information if True, log_output_file is the destination of a log file
# if log_output_file is not provided there is no file created
# if log is False, logging is turned off
pipeline_optimizer = PipelineOptimizer(
    data=data_reader,
    classifiers=[
        "AdaBoost",
        "Bagging",
        "MultiLayerPerceptron",
        "RandomForest",
        "ExtremelyRandomizedTrees",
        "LinearSVC",
    ],
    feature_selection_algorithms=[
        "SelectKBest",
        "SelectPercentile",
        "ParticleSwarmOptimization",
        "VarianceThreshold",
    ],
    feature_transform_algorithms=["Normalizer", "StandardScaler"],
    categorical_features_encoder="OneHotEncoder",
    imputer="SimpleImputer",
    log=True,
    log_verbose=True,
    log_output_file="output.log",
)

# runs the optimization process
# one of the possible pipelines in this case is: SelectPercentile -> Normalizer -> RandomForest
# returns the best found pipeline
# the chosen fitness function and optimization algorithm are Accuracy and Particle Swarm Algorithm
pipeline = pipeline_optimizer.run(
    "Accuracy", 10, 10, 30, 30, "ParticleSwarmAlgorithm", "ParticleSwarmAlgorithm"
)

# pipeline variable contains Pipeline object that can be used for further classification, exported as an object (that can be later loaded and used) or exported as text file
