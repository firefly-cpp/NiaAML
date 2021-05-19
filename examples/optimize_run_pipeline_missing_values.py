from niaaml import Pipeline
from niaaml.classifiers import MultiLayerPerceptron
from niaaml.preprocessing.feature_selection import VarianceThreshold
from niaaml.preprocessing.feature_transform import Normalizer
from niaaml.data import CSVDataReader
from niaaml.preprocessing.encoding import encode_categorical_features
from niaaml.preprocessing.imputation import impute_features
import os
import numpy
import pandas

"""
This example presents how to use the Pipeline class individually. You may use this if you want to test out a specific classification pipeline.
We use a dataset that contains categorical and numerical features with missing values.
"""

# prepare data reader using csv file
data_reader = CSVDataReader(
    src=os.path.dirname(os.path.abspath(__file__))
    + "/example_files/dataset_categorical_missing.csv",
    has_header=False,
    contains_classes=True,
)

features = data_reader.get_x()

# we use the utility method impute_features to get imputers for the features with missing values, but you may instantiate and fit
# imputers separately and pass them as a dictionary (as long as they are implemented as this framework suggests), with keys as column names or indices (if there is no header in the csv)
# there should be as many imputers as the features with missing values
# this example uses Simple Imputer
features, imputers = impute_features(features, "SimpleImputer")

# exactly the same goes for encoders
_, encoders = encode_categorical_features(features, "OneHotEncoder")

# instantiate a Pipeline object
pipeline = Pipeline(
    feature_selection_algorithm=VarianceThreshold(),
    feature_transform_algorithm=Normalizer(),
    classifier=MultiLayerPerceptron(),
    categorical_features_encoders=encoders,
    imputers=imputers,
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

# pipeline variable contains a Pipeline object that can be used for further classification, exported as an object (that can later be loaded and used) or exported as text file
