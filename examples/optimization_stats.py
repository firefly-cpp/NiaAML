from niaaml.utilities import OptimizationStats
import numpy as np

"""
In this example, we show how the OptimizationStats class can be used. Normally, it is used in the background when the Pipeline's optimize method is called.
You may also use it on its own if you find useful.
"""

# dummy array with expected results of classification process
y = np.array(['Class 1', 'Class 1', 'Class 1', 'Class 2', 'Class 1', 'Class 2',
'Class 2', 'Class 2', 'Class 2', 'Class 1', 'Class 1', 'Class 2',
'Class 1', 'Class 2', 'Class 1', 'Class 1', 'Class 1', 'Class 1',
'Class 2', 'Class 1'])

# dummy array with predicted classes
predicted = np.array(['Class 1', 'Class 1', 'Class 1', 'Class 2', 'Class 2', 'Class 2',
'Class 1', 'Class 1', 'Class 1', 'Class 2', 'Class 1', 'Class 1',
'Class 2', 'Class 2', 'Class 1', 'Class 2', 'Class 1', 'Class 2',
'Class 2', 'Class 2'])

# instantiate OptimizationStats
stats = OptimizationStats(predicted, y)

# print user-friendly text representation
print(stats.to_string())