import os
from niaaml import Pipeline

"""
In this example, we show how to load a saved Pipeline object from a file. You can use all of its methods after it's been successfully loaded.
"""

# load Pipeline object from a file
pipeline = Pipeline.load(os.path.dirname(os.path.abspath(__file__)) + '/example_files/pipeline.ppln')

# all of the Pipeline's classes methods can be called after a successful load