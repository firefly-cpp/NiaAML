from niaaml.utilities import OptimizationStats
import numpy as np

"""
In this example, we show how the OptimizationStats class can be used. Normally, it is used in the background when the Pipeline's optimize method is called.
You may also use it on its own if you find any use.
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

# let's say these are fitness scores of the 10-fold cross validation
fitness_scores = np.array([0.5, 0.55, 0.45, 0.57, 0.6, 0.47, 0.53, 0.52, 0.58, 0.44])

# instantiate OptimizationStats
# let's say the used fitness function's name is Accuracy
stats = OptimizationStats(predicted, y, fitness_scores, 'Accuracy')

# export boxplot of the 10-fold cross validation scores
stats.export_boxplot('boxplot.png')

# print user-friendly text representation
print(stats.to_string())