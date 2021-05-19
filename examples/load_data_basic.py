from niaaml.data import BasicDataReader
import numpy

"""
This example presents how to instantiate BasicDataReader and use its methods. You can use it to contain data in a single variable
or as an input to an instance of the PipelineOptimizer class.
"""

# BasicDataReader instance uses arrays on the input (x and y arrays)
data_reader = BasicDataReader(
    x=numpy.random.uniform(low=0.0, high=15.0, size=(50, 3)),
    y=numpy.random.choice(["Class 1", "Class 2"], size=50),
)

# get x and y arrays and print them
print(data_reader.get_x())
print(data_reader.get_y())
