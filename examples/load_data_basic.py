from niaaml.data import BasicDataReader
import numpy

data_reader = BasicDataReader(
    x=numpy.random.uniform(low=0.0, high=15.0, size=(50, 3)),
    y=numpy.random.choice(['Class 1', 'Class 2'], size=50)
)

print(data_reader.get_x())
print(data_reader.get_y())