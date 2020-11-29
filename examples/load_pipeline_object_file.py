import os
from niaaml import Pipeline

# load Pipeline object from a file
pipeline = Pipeline.load(os.path.dirname(os.path.abspath(__file__)) + '/example_files/pipeline.ppln')

# all of the Pipeline's classes methods can be called after a successful load