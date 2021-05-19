import os
from niaaml.data import CSVDataReader

"""
This example presents how to instantiate CSVDataReader and use its methods. You can use it to contain data in a single variable,
or as an input to an instance of the PipelineOptimizer class.
"""

# CSVDataReader gets a path to csv file on the input, reads and parses it into the x and y arrays
# has_header and contains_classes arguments needs to be set according to the input csv file's structure
data_reader = CSVDataReader(
    src=os.path.dirname(os.path.abspath(__file__)) + "/example_files/dataset.csv",
    has_header=False,
    contains_classes=True,
)

# get x and y arrays and print them
print(data_reader.get_x())
print(data_reader.get_y())
