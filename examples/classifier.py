from niaaml.classifiers import AdaBoost
import os
from niaaml.data import CSVDataReader
import numpy

"""
In this example, we show how to individually use an implemented classifier and its methods. In this case we use AdaBoost for demonstration, but
you can use any of the implemented classifiers in the same way.
"""

# prepare data reader using csv file
data_reader = CSVDataReader(
    src=os.path.dirname(os.path.abspath(__file__)) + "/example_files/dataset.csv",
    has_header=False,
    contains_classes=True,
)

# instantiate AdaBoost classifier
classifier = AdaBoost()

# set parameters of the classifier
classifier.set_parameters(n_estimators=50, algorithm="SAMME")

# fit classifier to the data
classifier.fit(data_reader.get_x(), data_reader.get_y())

# predict classes of the dummy input
predicted = classifier.predict(
    numpy.random.uniform(low=0.0, high=15.0, size=(30, data_reader.get_x().shape[1]))
)

# print classifier in a user-friendly form
print(classifier.to_string())
