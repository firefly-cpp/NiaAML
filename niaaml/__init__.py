from niaaml import classifiers
from niaaml import data
from niaaml import preprocessing
from niaaml import fitness
from niaaml.utilities import float_converter
from niaaml.utilities import MinMax
from niaaml.utilities import ParameterDefinition
from niaaml.utilities import Factory
from niaaml.pipeline_optimizer import PipelineOptimizer
from niaaml.pipeline import Pipeline
from niaaml.pipeline_component import PipelineComponent

__all__ = [
    'classifiers',
    'data',
    'preprocessing',
    'fitness',
    'float_converter',
    'get_bin_index',
    'MinMax',
    'ParameterDefinition',
    'OptimizationStats',
    'Factory',
    'PipelineOptimizer',
    'Pipeline',
    'PipelineComponent'
]
__project__ = 'niaaml'
__version__ = '0.1.0'
