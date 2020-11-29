from niaaml.classifiers import AdaBoost
import os
from niaaml.data import CSVDataReader
import numpy

data_reader = CSVDataReader(src=os.path.dirname(os.path.abspath(__file__)) + '/example_files/dataset.csv', has_header=False, contains_classes=True)

classifier = AdaBoost()

classifier.set_parameters(n_estimators=50, algorithm='SAMME')

classifier.fit(data_reader.get_x(), data_reader.get_y())
predicted = classifier.predict(numpy.random.uniform(low=0.0, high=15.0, size=(30,data_reader.get_x().shape[1])))

print(classifier.to_string())