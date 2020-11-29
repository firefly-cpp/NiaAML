import os
from niaaml import PipelineOptimizer, Pipeline
from niaaml.data import CSVDataReader

data_reader = CSVDataReader(src=os.path.dirname(os.path.abspath(__file__)) + '/example_files/dataset.csv', has_header=False, contains_classes=True)

pipeline_optimizer = PipelineOptimizer(
    data=data_reader,
    classifiers=['AdaBoost', 'Bagging', 'MultiLayerPerceptron', 'RandomForest', 'ExtremelyRandomizedTrees', 'LinearSVC'],
    feature_selection_algorithms=['SelectKBest', 'SelectPercentile', 'ParticleSwarmOptimization', 'VarianceThreshold'],
    feature_transform_algorithms=['Normalizer', 'StandardScaler']
)
pipeline = pipeline_optimizer.run('Accuracy', 20, 20, 400, 400, 'ParticleSwarmAlgorithm', 'ParticleSwarmAlgorithm')

# pipeline variable contains Pipeline object that can be used for further classification, exported as an object (that can be later loaded and used) or exported as text file