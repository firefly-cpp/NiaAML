import numpy as np
from niaaml.pipeline import Pipeline
from niaaml.classifiers import ClassifierFactory
from niaaml.feature_selection_algorithms import FeatureSelectionAlgorithmFactory
from niaaml.preprocessing_algorithms import PreprocessingAlgorithmFactory
from NiaPy.task import StoppingTask
from NiaPy.benchmarks import Benchmark
from NiaPy.algorithms.basic import ParticleSwarmOptimization

__all__ = [
    'PipelineOptimizer',
    'PipelineOptimizerBenchmark'
]

class PipelineOptimizer():
    r"""Optimization task that finds the best classification pipeline according to the given input.
    
	Date:
		2020

	Author
		Luka Pečnik

	License:
        MIT

	Attributes:
		__data (DataReader): Instance of any DataReader implementation.
		__feature_selection_algorithms (Iterable[FeatureSelectionAlgorithm]): Array of possible feature selection algorithms.
		__preprocessing_algorithms (Iterable[PreprocessingAlgorithm]): Array of possible preprocessing algorithms.
        __classifiers (Iterable[Classifier]): Array of possible classifiers.
		__pipelines_numeric (numpy.ndarray[float]): Numeric representation of pipelines.
		__pipelines (Iterable[Pipeline]): Actual pipelines.
        __pop_size (int): Number of individuals in the pipeline optimizer's population.

        __classifier_factory (ClassifierFactory): Factory for classifier instances.
        __preprocessing_algorithm_factory (ClassifierFactory): Factory for preprocessing algorithm instances.
        __feature_selection_algorithm_factory (ClassifierFactory): Factory for feature selection algorithm instances.
    """
    __data = None
    __feature_selection_algorithms = None
    __preprocessing_algorithms = None
    __classifiers = None

    __pop_size = None
    __pipelines_numeric = None
    __pipelines = None

    __classifier_factory = ClassifierFactory()
    __preprocessing_algorithm_factory = PreprocessingAlgorithmFactory()
    __feature_selection_algorithm_factory = FeatureSelectionAlgorithmFactory()

    def __init__(self, **kwargs):
        r"""Initialize task.
        """
        self._set_parameters(**kwargs)
        self.__initialize_population(self.__pop_size)
    
    def _set_parameters(self, data, feature_selection_algorithms, preprocessing_algorithms, classifiers, pop_size, **kwargs):
        r"""Set the parameters/arguments of the task.

		Arguments:
            data (DataReader): Instance of any DataReader implementation.
            feature_selection_algorithms (Iterable[FeatureSelectionAlgorithm]): Array of possible feature selection algorithms.
            preprocessing_algorithms (Iterable[PreprocessingAlgorithm]): Array of possible preprocessing algorithms.
            classifiers (Iterable[Classificator]): Array of possible classifiers.
            pop_size (int): Number of individuals in the pipeline optimizer's population.
        """
        self.__data = data

        self.__preprocessing_algorithms = preprocessing_algorithms
        try:
            self.__preprocessing_algorithms.index(None)
        except:
            self.__preprocessing_algorithms.insert(0, None)

        self.__classifiers = classifiers
        self.__feature_selection_algorithms = feature_selection_algorithms
        self.__pop_size = pop_size

    def __initialize_population(self, pop_size):
        r"""Initialize population of pipelines to find the best setup.

        Arguments:
            pop_size (int): Number of individuals.
        """

        self.__pipelines_numeric = np.random.uniform(size=(pop_size, 3))
        self.__pipelines = [
            Pipeline(
                data=self.__data,
                feature_selection_algorithm=self.__float_to_instance(i[0], self.__feature_selection_algorithms, self.__feature_selection_algorithm_factory) if self.__feature_selection_algorithms is not None and len(self.__feature_selection_algorithms) > 0 else None,
                preprocessing_algorithm=self.__float_to_instance(i[1], self.__preprocessing_algorithms, self.__preprocessing_algorithm_factory) if self.__preprocessing_algorithms is not None and len(self.__preprocessing_algorithms) > 0 else None,
                classifier=self.__float_to_instance(i[2], self.__classifiers, self.__classifier_factory)
                ) for i in self.__pipelines_numeric
        ]
    
    def __float_to_instance(self, value, collection, factory):
        r"""TODO
        """
        name = collection[np.int(np.round(value * (len(collection) - 1)))]
        return factory.get_result(name) if name is not None else None

    def run(self, pipeline_population_size, pipeline_classifier_population_size, number_of_pipeline_evaluations, number_of_classifier_evaluations):
        r"""TODO
        """
        algo = ParticleSwarmOptimization(NP=pipeline_population_size)    # TODO define InitPopFunc
        task = StoppingTask(D=3, nFES=number_of_pipeline_evaluations, benchmark=PipelineOptimizerBenchmark())
        best = algo.run(task)
        return best

class PipelineOptimizerBenchmark(Benchmark):
    def __init__(self):
        Benchmark.__init__(self, 0.0, 1.0)
    
    def function(self):
        # TODO
        def evaluate(D, sol):
            return 0.0
        return evaluate