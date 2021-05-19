from niaaml.preprocessing.encoding import OneHotEncoder, encode_categorical_features
import os
from niaaml.data import CSVDataReader

"""
This example presents how to use an implemented categorical feature encoder and its methods individually. In this case, we use OneHotEncoder for demonstration, but
you can use any of the implemented encoders in the same way.
"""

# prepare data reader using csv file
data_reader = CSVDataReader(
    src=os.path.dirname(os.path.abspath(__file__))
    + "/example_files/dataset_categorical.csv",
    has_header=False,
    contains_classes=True,
)

# instantiate OneHotEncoder
ohe = OneHotEncoder()

# fit, transform and print to output the categorical feature in the dataset (index 6)
features = data_reader.get_x()
ohe.fit(features[[6]])
f = ohe.transform(features[[6]])
print(f)

# if you need to get an array of encoders for all of the categorical features in a dataset (and transformed DataFrame of features), you may use the utility method encode_categorical_features
transformed_features, encoders = encode_categorical_features(features, "OneHotEncoder")
