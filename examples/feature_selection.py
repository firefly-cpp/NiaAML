from niaaml.preprocessing.feature_selection import SelectKBest
import os
from niaaml.data import CSVDataReader
from sklearn.feature_selection import chi2

# prepare data reader using csv file
data_reader = CSVDataReader(src=os.path.dirname(os.path.abspath(__file__)) + '/example_files/dataset.csv', has_header=False, contains_classes=True)

# instantiate SelectKBest feature selection algorithms
fs = SelectKBest()

# set parameters of the object
fs.set_parameters(k=4, score_func=chi2)

# select best features according to the SelectKBest algorithm (returns boolean mask of the selected features - True if selected, False if not)
features_mask = fs.select_features(data_reader.get_x(), data_reader.get_y())

# print feature selection algorithm in a user-friendly form
print(fs.to_string())