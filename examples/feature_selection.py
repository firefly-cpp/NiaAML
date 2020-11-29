from niaaml.preprocessing.feature_selection import SelectKBest
import os
from niaaml.data import CSVDataReader
from sklearn.feature_selection import chi2

data_reader = CSVDataReader(src=os.path.dirname(os.path.abspath(__file__)) + '/example_files/dataset.csv', has_header=False, contains_classes=True)

fs = SelectKBest()

fs.set_parameters(k=4, score_func=chi2)

features_mask = fs.select_features(data_reader.get_x(), data_reader.get_y())

print(fs.to_string())