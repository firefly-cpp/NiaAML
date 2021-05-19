from niaaml.preprocessing.feature_selection import ParticleSwarmOptimization
import os
from niaaml.data import CSVDataReader

"""
This example presents how to use implemented feature selection algorithms that use threshold mechanism.
"""

# prepare data reader using csv file
data_reader = CSVDataReader(
    src=os.path.dirname(os.path.abspath(__file__)) + "/example_files/dataset.csv",
    has_header=False,
    contains_classes=True,
)

# instantiate feature selection algorithm
fs = ParticleSwarmOptimization()
# BatAlgorithm, DifferentialEvolution, GreyWolfOptimizer and jDEFSTH also use threshold mechanism

# set parameters of the instantiated algorithm
fs.set_parameters(C1=1.5, C2=2.0)

# select best features according to the ParticleSwarmOptimization algorithm (returns boolean mask of the selected features - True if selected, False if not)
features_mask = fs.select_features(data_reader.get_x(), data_reader.get_y())

# print feature selection algorithm in a user-friendly form
print(fs.to_string())
