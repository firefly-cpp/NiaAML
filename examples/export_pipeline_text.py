from niaaml import Pipeline
from niaaml.classifiers import AdaBoost
from niaaml.preprocessing.feature_selection import SelectKBest
from niaaml.preprocessing.feature_transform import Normalizer

"""
In this example, we show how to export a pipeline object into a text file in a user-friendly form. Text file cannot be loaded back into a Python program in
a form of a Pipeline object.
"""

# instantiate a Pipeline object with AdaBoost classifier, SelectKBest feature selection algorithm and Normalizer as feature transformation algorithm
pipeline = Pipeline(
    feature_selection_algorithm=SelectKBest(),
    feature_transform_algorithm=Normalizer(),
    classifier=AdaBoost()
)

# export the object to a file in a user-friendly form
pipeline.export_text('exported_pipeline.txt')