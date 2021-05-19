from niaaml.fitness import Precision
from niaaml.data import CSVDataReader
import os
import numpy

"""
This example presents how to use an implemented fitness function and its method individually. In this case, we use Precision for demonstration, but
you can use any of the implemented fitness functions in the same way.
"""

# prepare data reader using csv file
data_reader = CSVDataReader(
    src=os.path.dirname(os.path.abspath(__file__)) + "/example_files/dataset.csv",
    has_header=False,
    contains_classes=True,
)

# lets say the following array contains predictions after the classification process
predictions = numpy.random.choice(
    ["Class 1", "Class 2"], size=data_reader.get_y().shape
)

# instantiate instance of a fitness function (Precision in this case)
fitness_func = Precision()

# calculate fitness value
precision = fitness_func.get_fitness(predictions, data_reader.get_y())

# precision will probably be low due to dummy data
print(precision)
