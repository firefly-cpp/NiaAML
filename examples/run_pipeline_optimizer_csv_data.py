import os
from niaaml import PipelineOptimizer, Pipeline
from niaaml.data import CSVDataReader

# prepare data reader using csv file
data_reader = CSVDataReader(src=os.path.dirname(os.path.abspath(__file__)) + '/example_files/dataset.csv', has_header=False, contains_classes=True)

# instantiate PipelineOptimizer that chooses among specified classifiers, feature selection algorithms and feature transform algorithms
pipeline_optimizer = PipelineOptimizer(
    data=data_reader,
    classifiers=['AdaBoost', 'Bagging', 'MultiLayerPerceptron', 'RandomForest', 'ExtremelyRandomizedTrees', 'LinearSVC'],
    feature_selection_algorithms=['SelectKBest', 'SelectPercentile', 'ParticleSwarmOptimization', 'VarianceThreshold'],
    feature_transform_algorithms=['Normalizer', 'StandardScaler']
)

# runs the optimization process
# one of the possible pipelines in this case is: SelectPercentile -> Normalizer -> RandomForest
# returns the best found pipeline
# the chosen fitness function and optimization algorithm are Accuracy and Particle Swarm Algorithm
pipeline = pipeline_optimizer.run('Accuracy', 20, 20, 400, 400, 'ParticleSwarmAlgorithm', 'ParticleSwarmAlgorithm')

# pipeline variable contains Pipeline object that can be used for further classification, exported as an object (that can be later loaded and used) or exported as text file