from niaaml.preprocessing.feature_transform import Normalizer
import os
from niaaml.data import CSVDataReader

# prepare data reader using csv file
data_reader = CSVDataReader(src=os.path.dirname(os.path.abspath(__file__)) + '/example_files/dataset.csv', has_header=False, contains_classes=True)

# instantiate Normalizer
ft = Normalizer()

# set parameters of the Normalizer
ft.set_parameters(norm='l2')

# fit the algorithm to the input data
ft.fit(data_reader.get_x())

# transform features
transformed_features = ft.transform(data_reader.get_x())

# print feature transform algorithm in a user-friendly form
print(ft.to_string())