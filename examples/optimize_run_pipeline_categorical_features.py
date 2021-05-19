from niaaml import Pipeline
from niaaml.classifiers import MultiLayerPerceptron
from niaaml.preprocessing.feature_selection import VarianceThreshold
from niaaml.preprocessing.feature_transform import Normalizer
from niaaml.data import CSVDataReader
from niaaml.preprocessing.encoding import encode_categorical_features
import os
import numpy
import pandas

"""
This example presents how to use the Pipeline class individually. You may use this if you want to test out a specific classification pipeline.
We use a dataset that contains categorical and numerical features.
"""

# prepare data reader using csv file
data_reader = CSVDataReader(
    src=os.path.dirname(os.path.abspath(__file__))
    + "/example_files/dataset_categorical.csv",
    has_header=False,
    contains_classes=True,
)

# we use the utility method encode_categorical_features to get encoders for the categorical features, but you may instantiate and fit
# feature encoders separately and pass them as an array (as long as they are implemented as this framework suggests)
# there should be as many encoders as categorical features
# this example uses One-Hot Encoding
_, encoders = encode_categorical_features(data_reader.get_x(), "OneHotEncoder")

# instantiate a Pipeline object
pipeline = Pipeline(
    feature_selection_algorithm=VarianceThreshold(),
    feature_transform_algorithm=Normalizer(),
    classifier=MultiLayerPerceptron(),
    categorical_features_encoders=encoders,
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
        [
            [
                10.32440339,
                3.195964543,
                1.215275549,
                3.741461311,
                11.6736581,
                6.435247906,
                "a",
            ]
        ]
    )
)

# pipeline variable contains a Pipeline object that can be used for further classification, exported as an object (that can later be loaded and used) or exported as a text file
