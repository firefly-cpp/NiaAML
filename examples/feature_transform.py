from niaaml.preprocessing.feature_transform import Normalizer
import os
from niaaml.data import CSVDataReader

data_reader = CSVDataReader(src=os.path.dirname(os.path.abspath(__file__)) + '/example_files/dataset.csv', has_header=False, contains_classes=True)

ft = Normalizer()

ft.set_parameters(norm='l2')

ft.fit(data_reader.get_x())
transformed_features = ft.transform(data_reader.get_x())

print(ft.to_string())