import numpy as np
from niaaml.pipeline import Pipeline
from niaaml.classifiers import ClassifierFactory
from niaaml.feature_selection_algorithms import FeatureSelectionAlgorithmFactory
from niaaml.preprocessing_algorithms import PreprocessingAlgorithmFactory
from NiaPy.task import StoppingTask, OptimizationType
from NiaPy.benchmarks import Benchmark
from NiaPy.algorithms.utility import AlgorithmUtility

__all__ = [
    'PipelineOptimizer'
]

def _initialize_population(task, NP, rnd=np.random, **kwargs):
    r"""TODO
    """
    pop = np.random.uniform(size=(NP, 3))
    fpop = np.apply_along_axis(task.eval, 1, pop)
    return pop, fpop

class PipelineOptimizer:
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

        __classifier_factory (ClassifierFactory): Factory for classifier instances.
        __preprocessing_algorithm_factory (ClassifierFactory): Factory for preprocessing algorithm instances.
        __feature_selection_algorithm_factory (ClassifierFactory): Factory for feature selection algorithm instances.

        __optimization_algorithm (str): Name of the optimization algorithm to use.
        __niapy_algorithm_utility (AlgorithmUtility): Utility class used to get an optimization algorithm.
    """
    __data = None
    __feature_selection_algorithms = None
    __preprocessing_algorithms = None
    __classifiers = None

    __pipelines_numeric = None
    __pipelines = None

    __optimization_algorithm = None
    __niapy_algorithm_utility = AlgorithmUtility()

    def __init__(self, **kwargs):
        r"""Initialize task.
        """
        self._set_parameters(**kwargs)
    
    def _set_parameters(self, data, feature_selection_algorithms, preprocessing_algorithms, classifiers, optimization_algorithm, **kwargs):
        r"""Set the parameters/arguments of the task.

		Arguments:
            data (DataReader): Instance of any DataReader implementation.
            feature_selection_algorithms (Iterable[FeatureSelectionAlgorithm]): Array of possible feature selection algorithms.
            preprocessing_algorithms (Iterable[PreprocessingAlgorithm]): Array of possible preprocessing algorithms.
            classifiers (Iterable[Classificator]): Array of possible classifiers.
            optimization_algorithm (str): Name of the optimization algorithm to use.
        """
        self.__data = data
        self.__optimization_algorithm = optimization_algorithm

        self.__preprocessing_algorithms = preprocessing_algorithms
        if self.__preprocessing_algorithms is not None:
            try:
                self.__preprocessing_algorithms.index(None)
            except:
                self.__preprocessing_algorithms.insert(0, None)

        self.__classifiers = classifiers
        self.__feature_selection_algorithms = feature_selection_algorithms

    def run(self, pipeline_population_size, classifier_population_size, number_of_pipeline_evaluations, number_of_classifier_evaluations):
        r"""TODO
        """
        algo = self.__niapy_algorithm_utility.get_algorithm(self.__optimization_algorithm)
        algo.NP = pipeline_population_size
        algo.InitPopFunc = _initialize_population

        task = StoppingTask(
            D=3,
            nFES=number_of_pipeline_evaluations,
            benchmark=_PipelineOptimizerBenchmark(self.__data, self.__feature_selection_algorithms, self.__preprocessing_algorithms, self.__classifiers, classifier_population_size, number_of_classifier_evaluations),
            optType=OptimizationType.MAXIMIZATION
            )
        best = algo.run(task)
        return best

class _PipelineOptimizerBenchmark(Benchmark):
    r"""TODO
    """
    __data = None
    __feature_selection_algorithms = None
    __preprocessing_algorithms = None
    __classifiers = None

    __classifier_factory = ClassifierFactory()
    __preprocessing_algorithm_factory = PreprocessingAlgorithmFactory()
    __feature_selection_algorithm_factory = FeatureSelectionAlgorithmFactory()

    def __init__(self, data, feature_selection_algorithms, preprocessing_algorithms, classifiers, classifier_population_size, number_of_classifier_evaluations):
        r"""TODO
        """
        self.__data = data
        self.__feature_selection_algorithms = feature_selection_algorithms
        self.__preprocessing_algorithms = preprocessing_algorithms
        self.__classifiers = classifiers
        self.__classifier_population_size = classifier_population_size
        self.__number_of_classifier_evaluations = number_of_classifier_evaluations
        Benchmark.__init__(self, 0.0, 1.0)

    def __float_to_instance(self, value, collection, factory):
        r"""TODO
        """
        name = collection[np.int(np.round(value * (len(collection) - 1)))]
        return factory.get_result(name) if name is not None else None
    
    def function(self):
        r"""TODO
        """
        # TODO
        def evaluate(D, sol):
            r"""TODO
            """
            pipeline = Pipeline(
                data=self.__data,
                feature_selection_algorithm=self.__float_to_instance(sol[0], self.__feature_selection_algorithms, self.__feature_selection_algorithm_factory) if self.__feature_selection_algorithms is not None and len(self.__feature_selection_algorithms) > 0 else None,
                preprocessing_algorithm=self.__float_to_instance(sol[1], self.__preprocessing_algorithms, self.__preprocessing_algorithm_factory) if self.__preprocessing_algorithms is not None and len(self.__preprocessing_algorithms) > 0 else None,
                classifier=self.__float_to_instance(sol[2], self.__classifiers, self.__classifier_factory)
            )
            return pipeline.optimize(self.__classifier_population_size, self.__number_of_classifier_evaluations)
        
        return evaluate