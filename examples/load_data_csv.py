import os
from niaaml.data import CSVDataReader

data_reader = CSVDataReader(src=os.path.dirname(os.path.abspath(__file__)) + '/example_files/dataset.csv', has_header=False, contains_classes=True)

print(data_reader.get_x())
print(data_reader.get_y())