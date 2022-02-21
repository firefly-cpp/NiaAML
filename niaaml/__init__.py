from niaaml import classifiers
from niaaml import data
from niaaml import preprocessing
from niaaml import fitness
from niaaml.utilities import MinMax
from niaaml.utilities import ParameterDefinition
from niaaml.utilities import Factory
from niaaml.utilities import OptimizationStats
from niaaml.utilities import get_bin_index
from niaaml.pipeline_optimizer import PipelineOptimizer
from niaaml.pipeline import Pipeline
from niaaml.pipeline_component import PipelineComponent
from niaaml.logger import Logger

__all__ = [
    "classifiers",
    "data",
    "preprocessing",
    "fitness",
    "get_bin_index",
    "MinMax",
    "ParameterDefinition",
    "OptimizationStats",
    "Factory",
    "PipelineOptimizer",
    "Pipeline",
    "PipelineComponent",
    "Logger",
]

__project__ = "niaaml"
__version__ = "1.1.7"
