from niaaml import PipelineOptimizer
from niaaml.data import BasicDataReader
import numpy

"""
This example presents how to use the PipelineOptimizer class. This example is using an instance of BasicDataReader.
The instantiated PipelineOptimizer try to compose the best pipeline with the components that are specified in its constructor.
"""

# prepare data reader using features and classes from arrays
# in this case random dummy arrays are generated
data_reader = BasicDataReader(
    x=numpy.random.uniform(low=0.0, high=15.0, size=(50, 3)),
    y=numpy.random.choice(["Class 1", "Class 2"], size=50),
)

# instantiate PipelineOptimizer that chooses among specified classifiers, feature selection algorithms and feature transform algorithms
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
