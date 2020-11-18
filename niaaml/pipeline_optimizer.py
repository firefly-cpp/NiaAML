import numpy as np
from niaaml.pipeline import Pipeline
from niaaml.classifiers import ClassifierFactory
from niaaml.preprocessing.feature_selection import FeatureSelectionAlgorithmFactory
from niaaml.preprocessing.feature_transform import FeatureTransformAlgorithmFactory
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
		__feature_selection_algorithms (Iterable[str]): Array of names of possible feature selection algorithms.
		__feature_transform_algorithms (Iterable[str]): Array of names of possible feature transform algorithms.
        __classifiers (Iterable[Classifier]): Array of names of possible classifiers.
		__pipelines_numeric (numpy.ndarray[float]): Numeric representation of pipelines.

        __optimization_algorithm (str): Name of the optimization algorithm to use.
        __niapy_algorithm_utility (AlgorithmUtility): Utility class used to get an optimization algorithm.
    """

    def __init__(self, **kwargs):
        r"""Initialize task.
        """
        self.__data = None
        self.__feature_selection_algorithms = None
        self.__feature_transform_algorithms = None
        self.__classifiers = None
        self.__pipelines_numeric = None
        self.__optimization_algorithm = None
        self.__niapy_algorithm_utility = AlgorithmUtility()

        self._set_parameters(**kwargs)
    
    def _set_parameters(self, data, feature_selection_algorithms, feature_transform_algorithms, classifiers, optimization_algorithm, **kwargs):
        r"""Set the parameters/arguments of the task.

		Arguments:
            data (DataReader): Instance of any DataReader implementation.
            feature_selection_algorithms (Iterable[str]): Array of names of possible feature selection algorithms.
            feature_transform_algorithms (Iterable[str]): Array of names of possible feature transform algorithms.
            classifiers (Iterable[Classificator]): Array of names of possible classifiers.
            optimization_algorithm (str): Name of the optimization algorithm to use.
        """
        self.__data = data
        self.__optimization_algorithm = optimization_algorithm

        self.__feature_transform_algorithms = feature_transform_algorithms
        if self.__feature_transform_algorithms is not None:
            try:
                self.__feature_transform_algorithms.index(None)
            except:
                self.__feature_transform_algorithms.insert(0, None)

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
            benchmark=_PipelineOptimizerBenchmark(self.__data, self.__feature_selection_algorithms, self.__feature_transform_algorithms, self.__classifiers, classifier_population_size, number_of_classifier_evaluations),
            optType=OptimizationType.MAXIMIZATION
            )
        best = algo.run(task)
        return best

class _PipelineOptimizerBenchmark(Benchmark):
    r"""TODO
    """
    __classifier_factory = ClassifierFactory()
    __feature_transform_algorithm_factory = FeatureTransformAlgorithmFactory()
    __feature_selection_algorithm_factory = FeatureSelectionAlgorithmFactory()

    def __init__(self, data, feature_selection_algorithms, feature_transform_algorithms, classifiers, classifier_population_size, number_of_classifier_evaluations):
        r"""TODO
        """
        self.__data = data
        self.__feature_selection_algorithms = feature_selection_algorithms
        self.__feature_transform_algorithms = feature_transform_algorithms
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
        def evaluate(D, sol):
            r"""TODO
            """
            pipeline = Pipeline(
                data=self.__data,
                feature_selection_algorithm=self.__float_to_instance(sol[0], self.__feature_selection_algorithms, self.__feature_selection_algorithm_factory) if self.__feature_selection_algorithms is not None and len(self.__feature_selection_algorithms) > 0 else None,
                feature_transform_algorithms=self.__float_to_instance(sol[1], self.__feature_transform_algorithms, self.__feature_transform_algorithm_factory) if self.__feature_transform_algorithms is not None and len(self.__feature_transform_algorithms) > 0 else None,
                classifier=self.__float_to_instance(sol[2], self.__classifiers, self.__classifier_factory)
            )
            return pipeline.optimize(self.__classifier_population_size, self.__number_of_classifier_evaluations)
        
        return evaluate