from niaaml.preprocessing.imputation import SimpleImputer, impute_features
import os
from niaaml.data import CSVDataReader

"""
This example presents how to use an implemented missing features' imputer and its methods individually. In this case, we use SimpleImputer for demonstration, but
you can use any of the implemented imputers in the same way.
"""

# prepare data reader using csv file
data_reader = CSVDataReader(
    src=os.path.dirname(os.path.abspath(__file__))
    + "/example_files/dataset_categorical_missing.csv",
    has_header=False,
    contains_classes=True,
)

# instantiate SimpleImputer
si = SimpleImputer()

# fit, transform and print to output the feature in the dataset (index 6)
features = data_reader.get_x()
si.fit(features[[6]])
f = si.transform(features[[6]])
print(f)

# if you wish to get array of imputers for all of the features with missing values in a dataset (and transformed DataFrame of features), you may use the utility method impute_features
transformed_features, imputers = impute_features(features, "SimpleImputer")
