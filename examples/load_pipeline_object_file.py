import os
from niaaml import Pipeline

pipeline = Pipeline.load(os.path.dirname(os.path.abspath(__file__)) + '/example_files/pipeline.ppln')