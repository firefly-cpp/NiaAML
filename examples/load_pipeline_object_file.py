import os
from niaaml import Pipeline

"""
This example presents how to load a saved Pipeline object from a file. You can use all of its methods after it has been loaded successfully.
"""

# load Pipeline object from a file
pipeline = Pipeline.load(
    os.path.dirname(os.path.abspath(__file__)) + "/example_files/pipeline.ppln"
)

# all of the Pipeline's classes methods can be called after a successful load
