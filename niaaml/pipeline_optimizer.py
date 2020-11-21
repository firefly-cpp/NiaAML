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
		__feature_selection_algorithms (Optional[Iterable[str]]): Array of names of possible feature selection algorithms.
		__feature_transform_algorithms (Optional[Iterable[str]]): Array of names of possible feature transform algorithms.
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
    
    def _set_parameters(self, data, classifiers, optimization_algorithm, feature_selection_algorithms = None, feature_transform_algorithms = None, **kwargs):
        r"""Set the parameters/arguments of the task.

		Arguments:
            data (DataReader): Instance of any DataReader implementation.
            feature_selection_algorithms (Optional[Iterable[str]]): Array of names of possible feature selection algorithms.
            feature_transform_algorithms (Optional[Iterable[str]]): Array of names of possible feature transform algorithms.
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
    
    def get_data(self):
        return self.__data

    def get_feature_selection_algorithms(self):
        return self.__feature_selection_algorithms

    def get_feature_transform_algorithms(self):
        return self.__feature_transform_algorithms

    def get_classifiers(self):
        return self.__classifiers

    def run(self, pipeline_population_size, inner_population_size, number_of_pipeline_evaluations, number_of_inner_evaluations):
        r"""Run classification pipeline optimization process.

		Arguments:
            pipeline_population_size (uint): Number of pipeline individuals in the optimization process.
            inner_population_size (uint): Number of individuals in the hiperparameter optimization process.
            number_of_pipeline_evaluations (uint): Number of maximum evaluations.
            number_of_inner_evaluations (uint): Number of maximum inner evaluations.
        """

        def _initialize_population(task, NP, rnd=np.random, **kwargs):
            r"""NiaPy's InitPopFunc implementation.

            Arguments:
                task (NiaPy.task.Task): Implementation of NiaPy's Task class.
                NP (uint): Population size.
                rnd (any): Random number generator.
            
            Returns:
                Tuple[numpy.ndarray, numpy.ndarray[float]]]
            """
            pop = np.random.uniform(size=(NP, 3))
            fpop = np.apply_along_axis(task.eval, 1, pop)
            return pop, fpop

        algo = self.__niapy_algorithm_utility.get_algorithm(self.__optimization_algorithm)
        algo.NP = pipeline_population_size
        algo.InitPopFunc = _initialize_population

        task = StoppingTask(
            D=3,
            nFES=number_of_pipeline_evaluations,
            benchmark=self._PipelineOptimizerBenchmark(self, inner_population_size, number_of_inner_evaluations),
            optType=OptimizationType.MAXIMIZATION
            )
        best = algo.run(task)
        return best

    class _PipelineOptimizerBenchmark(Benchmark):
        r"""NiaPy Benchmark class implementation.

        Attributes:
            __inner_population_size (uint): Number of individuals in the hiperparameter optimization process.
            __number_of_inner_evaluations (uint): Number of maximum inner evaluations.

            __classifier_factory (ClassifierFactory): Factory for classifiers.
            __feature_transform_algorithm_factory (FeatureTransformAlgorithmFactory): Factory for feature transform algorithms.
            __feature_selection_algorithm_factory (FeatureSelectionAlgorithmFactory): Factory for feature selection algorithms.
        """
        __classifier_factory = ClassifierFactory()
        __feature_transform_algorithm_factory = FeatureTransformAlgorithmFactory()
        __feature_selection_algorithm_factory = FeatureSelectionAlgorithmFactory()

        def __init__(self, parent, inner_population_size, number_of_inner_evaluations):
            r"""Initialize pipeline optimizer benchmark.

            Arguments:
                parent (PipelineOptimizer): Parent instance of PipelineOptimizer.
                inner_population_size (uint): Number of individuals in the hiperparameter optimization process.
                number_of_inner_evaluations (uint): Number of maximum inner evaluations.
            """
            self.__parent = parent
            self.__inner_population_size = inner_population_size
            self.__number_of_inner_evaluations = number_of_inner_evaluations
            Benchmark.__init__(self, 0.0, 1.0)

        def __float_to_instance(self, value, collection, factory):
            r"""Get instance of object from collection using factory.

            Arguments:
                value (float): Value to map.
                collection (Iterable[str]): Array of names of possible feature selection algorithms.
                factory (Factory): Implementation of the Factory class.
            
            Returns:
                PipelineComponent: Randomly initialized PipelineComponent instance.
            """
            bin_index = np.int(np.floor(value / (1.0 / len(collection))))
            if bin_index == len(collection):
                bin_index -= 1

            name = collection[bin_index]
            return factory.get_result(name) if name is not None else None
        
        def function(self):
            r"""Override Benchmark function.

            Returns:
                Callable[[int, Iterable[float]], float]: Fitness evaluation function.
            """
            def evaluate(D, sol):
                r"""Evaluate pipeline.

                Arguments:
                    D (uint): Number of dimensionas.
                    sol (Iterable[float]): Individual of population/ possible solution.
                
                Returns:
                    float: Fitness.
                """
                pipeline = Pipeline(
                    data=self.__parent.get_data(),
                    feature_selection_algorithm=self.__float_to_instance(sol[0], self.__parent.get_feature_selection_algorithms(), self.__feature_selection_algorithm_factory) if self.__parent.get_feature_selection_algorithms() is not None and len(self.__parent.get_feature_selection_algorithms()) > 0 else None,
                    feature_transform_algorithm=self.__float_to_instance(sol[1], self.__parent.get_feature_transform_algorithms(), self.__feature_transform_algorithm_factory) if self.__parent.get_feature_transform_algorithms() is not None and len(self.__parent.get_feature_transform_algorithms()) > 0 else None,
                    classifier=self.__float_to_instance(sol[2], self.__parent.get_classifiers(), self.__classifier_factory)
                )
                return pipeline.optimize(self.__inner_population_size, self.__number_of_inner_evaluations)
            
            return evaluate