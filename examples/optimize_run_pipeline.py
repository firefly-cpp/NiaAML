from niaaml import Pipeline
from niaaml.classifiers import MultiLayerPerceptron
from niaaml.preprocessing.feature_selection import VarianceThreshold
from niaaml.preprocessing.feature_transform import Normalizer
from niaaml.data import CSVDataReader
import os
import numpy
import pandas

"""
This example presents how to use the Pipeline class individually. You may use this if you want to test out a specific classification pipeline.
"""

# prepare data reader using csv file
data_reader = CSVDataReader(
    src=os.path.dirname(os.path.abspath(__file__)) + "/example_files/dataset.csv",
    has_header=False,
    contains_classes=True,
)

# instantiate a Pipeline object
pipeline = Pipeline(
    feature_selection_algorithm=VarianceThreshold(),
    feature_transform_algorithm=Normalizer(),
    classifier=MultiLayerPerceptron(),
)

# run pipeline optimization process (returns fitness value, but sets the best parameters for classifier, feature selection algorithm and feature transform algorithm during the process)
pipeline.optimize(
    data_reader.get_x(),
    data_reader.get_y(),
    10,
    50,
    "ParticleSwarmAlgorithm",
    "Accuracy",
)

# run the pipeline using dummy data
# you could run the pipeline before the optimization process, but get wrong predictions as nothing in the pipeline is fit for the given dataset
predicted = pipeline.run(
    pandas.DataFrame(
        numpy.random.uniform(
            low=0.0, high=15.0, size=(30, data_reader.get_x().shape[1])
        )
    )
)

# pipeline variable contains Pipeline object that can be used for further classification, exported as an object (that can be later loaded and used) or exported as text file
