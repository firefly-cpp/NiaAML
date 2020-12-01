import os
from niaaml import Pipeline
from niaaml.data import CSVDataReader
from niaaml.classifiers import AdaBoost
from niaaml.preprocessing.feature_selection import SelectKBest
from niaaml.preprocessing.feature_transform import Normalizer

"""
In this example, we show how to export a pipeline's 10-fold cross validation results into an image file in a form of box plot.
Optimization process needs to be run so the optimization statistics is available.
"""

# prepare data reader using csv file
data_reader = CSVDataReader(src=os.path.dirname(os.path.abspath(__file__)) + '/example_files/dataset.csv', has_header=False, contains_classes=True)

# instantiate a Pipeline object with AdaBoost classifier, SelectKBest feature selection algorithm and Normalizer as feature transformation algorithm
pipeline = Pipeline(
    feature_selection_algorithm=SelectKBest(),
    feature_transform_algorithm=Normalizer(),
    classifier=AdaBoost()
)

pipeline.optimize(data_reader.get_x(), data_reader.get_y(), 10, 50, 'ParticleSwarmAlgorithm', 'Accuracy')

# export the box plot
pipeline.export_boxplot('boxplot.png')