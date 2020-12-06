import numpy as np
from niaaml.pipeline import Pipeline
from niaaml.classifiers import ClassifierFactory
from niaaml.preprocessing.feature_selection import FeatureSelectionAlgorithmFactory
from niaaml.preprocessing.feature_transform import FeatureTransformAlgorithmFactory
from NiaPy.task import StoppingTask
from NiaPy.benchmarks import Benchmark
from NiaPy.algorithms.utility import AlgorithmUtility
from niaaml.utilities import get_bin_index
from niaaml.preprocessing.encoding.utility import encode_categorical_features
import pandas as pd

__all__ = [
    'PipelineOptimizer'
]

class PipelineOptimizer:
    r"""Optimization task that finds the best classification pipeline according to the given input.
    
    Date:
        2020

    Author:
        Luka Pečnik

    License:
        MIT

    Attributes:
        __data (DataReader): Instance of any DataReader implementation.
        __feature_selection_algorithms (Optional[Iterable[str]]): Array of names of possible feature selection algorithms.
        __feature_transform_algorithms (Optional[Iterable[str]]): Array of names of possible feature transform algorithms.
        __classifiers (Iterable[Classifier]): Array of names of possible classifiers.
        __categorical_features_encoder (str): Name of the encoder used for categorical features.
        __categorical_features_encoders (Iterable[FeatureEncoder]): Actual instances of FeatureEncoder for all categorical features.

        __niapy_algorithm_utility (AlgorithmUtility): Utility class used to get an optimization algorithm.
    """

    def __init__(self, **kwargs):
        r"""Initialize task.
        """
        self.__data = None
        self.__feature_selection_algorithms = None
        self.__feature_transform_algorithms = None
        self.__classifiers = None
        self.__categorical_features_encoder = None
        self.__categorical_features_encoders = None
        self.__niapy_algorithm_utility = AlgorithmUtility()

        self._set_parameters(**kwargs)
    
    def _set_parameters(self, data, classifiers, feature_selection_algorithms = None, feature_transform_algorithms = None, categorical_features_encoder = None, **kwargs):
        r"""Set the parameters/arguments of the task.

        Arguments:
            data (DataReader): Instance of any DataReader implementation.
            feature_selection_algorithms (Optional[Iterable[str]]): Array of names of possible feature selection algorithms.
            feature_transform_algorithms (Optional[Iterable[str]]): Array of names of possible feature transform algorithms.
            classifiers (Iterable[Classifier]): Array of names of possible classifiers.
            categorical_features_encoder (Optional[str]): Name of the encoder used for categorical features.
        """
        self.__data = data

        self.__feature_transform_algorithms = feature_transform_algorithms
        if self.__feature_transform_algorithms is not None:
            try:
                self.__feature_transform_algorithms.index(None)
            except:
                self.__feature_transform_algorithms.insert(0, None)

        self.__classifiers = classifiers
        self.__feature_selection_algorithms = feature_selection_algorithms
        self.__categorical_features_encoder = categorical_features_encoder
    
    def get_data(self):
        r"""Get data.

        Returns:
            DataReader: Instance of DataReader object.
        """
        return self.__data

    def get_feature_selection_algorithms(self):
        r"""Get feature selection algorithms.

        Returns:
            Iterable[str]: Feature selection algorithm names or None.
        """
        return self.__feature_selection_algorithms

    def get_feature_transform_algorithms(self):
        r"""Get feature transform algorithms.

        Returns:
            Iterable[str]: Feature transform algorithm names or None.
        """
        return self.__feature_transform_algorithms

    def get_classifiers(self):
        r"""Get classifiers.

        Returns:
            Iterable[str]: Classifier names.
        """
        return self.__classifiers

    def run(self, fitness_name, pipeline_population_size, inner_population_size, number_of_pipeline_evaluations, number_of_inner_evaluations, optimization_algorithm, inner_optimization_algorithm = None):
        r"""Run classification pipeline optimization process.

        Arguments:
            fitness_name (str): Name of the fitness class to use as a function.
            pipeline_population_size (uint): Number of pipeline individuals in the optimization process.
            inner_population_size (uint): Number of individuals in the hiperparameter optimization process.
            number_of_pipeline_evaluations (uint): Number of maximum evaluations.
            number_of_inner_evaluations (uint): Number of maximum inner evaluations.
            optimization_algorithm (str): Name of the optimization algorithm to use.
            inner_optimization_algorithm (Optional[str]): Name of the inner optimization algorithm to use. Defaults to the optimization_algorithm argument.
        
        Returns:
            Pipeline: Best pipeline found in the optimization process.
        """

        algo = self.__niapy_algorithm_utility.get_algorithm(optimization_algorithm)
        algo.NP = pipeline_population_size

        if self.__categorical_features_encoder is not None:
            features = self.__data.get_x()
            features, self.__categorical_features_encoders = encode_categorical_features(features, self.__categorical_features_encoder)
            self.__data.set_x(features)

        benchmark = _PipelineOptimizerBenchmark(self, fitness_name, inner_population_size, number_of_inner_evaluations, inner_optimization_algorithm if inner_optimization_algorithm is not None else optimization_algorithm)
        task = StoppingTask(
            D=3,
            nFES=number_of_pipeline_evaluations,
            benchmark=benchmark
            )
        algo.run(task)
        
        pipeline = benchmark.get_pipeline()
        if pipeline is not None:
            pipeline.set_categorical_features_encoders(self.__categorical_features_encoders)

        return pipeline

class _PipelineOptimizerBenchmark(Benchmark):
    r"""NiaPy Benchmark class implementation.
    
    Date:
        2020

    Author:
        Luka Pečnik

    Attributes:
        __parent (PipelineOptimizer): Parent instance of PipelineOptimizer.
        __inner_population_size (uint): Number of individuals in the hiperparameter optimization process.
        __number_of_inner_evaluations (uint): Number of maximum inner evaluations.
        __optimization_algorithm (str): Name of the optimization algorithm to use.
        __current_best_fitness (float): Current best fitness of the optimization process.
        __current_best_pipeline (Pipeline): Current best pipeline of the optimization process.
        __fitness_name (str): Name of the fitness class to use as a function.

        __classifier_factory (ClassifierFactory): Factory for classifiers.
        __feature_transform_algorithm_factory (FeatureTransformAlgorithmFactory): Factory for feature transform algorithms.
        __feature_selection_algorithm_factory (FeatureSelectionAlgorithmFactory): Factory for feature selection algorithms.
    """
    __classifier_factory = ClassifierFactory()
    __feature_transform_algorithm_factory = FeatureTransformAlgorithmFactory()
    __feature_selection_algorithm_factory = FeatureSelectionAlgorithmFactory()

    def __init__(self, parent, fitness_name, inner_population_size, number_of_inner_evaluations, inner_optimization_algorithm):
        r"""Initialize pipeline optimizer benchmark.

        Arguments:
            parent (PipelineOptimizer): Parent instance of PipelineOptimizer.
            fitness_name (str): Name of the fitness class to use as a function.
            inner_population_size (uint): Number of individuals in the hiperparameter optimization process.
            number_of_inner_evaluations (uint): Number of maximum inner evaluations.
            inner_optimization_algorithm (str): Name of the optimization algorithm to use.
        """
        self.__parent = parent
        self.__inner_population_size = inner_population_size
        self.__number_of_inner_evaluations = number_of_inner_evaluations
        self.__optimization_algorithm = inner_optimization_algorithm
        self.__current_best_fitness = float('inf')
        self.__current_best_pipeline = None
        self.__fitness_name = fitness_name
        Benchmark.__init__(self, 0.0, 1.0)

    def __float_to_instance(self, value, collection, factory):
        r"""Get instance of object from collection using factory.

        Arguments:
            value (float): Value to map.
            collection (Iterable[str]): Array of names of possible feature selection algorithms.
            factory (Factory): Implementation of the Factory class.
        
        Returns:
            PipelineComponent: New PipelineComponent instance.
        """
        bin_index = get_bin_index(value, len(collection))

        name = collection[bin_index]
        return factory.get_result(name) if name is not None else None
    
    def get_pipeline(self):
        r"""Get best pipeline found.

        Returns:
            Pipeline: Best pipeline found.
        """
        return self.__current_best_pipeline
    
    def function(self):
        r"""Override Benchmark function.

        Returns:
            Callable[[int, numpy.ndarray[float]], float]: Fitness evaluation function.
        """
        def evaluate(D, sol):
            r"""Evaluate pipeline.

            Arguments:
                D (uint): Number of dimensionas.
                sol (numpy.ndarray[float]): Individual of population/ possible solution.
            
            Returns:
                float: Fitness.
            """
            data = self.__parent.get_data()
            pipeline = Pipeline(
                feature_selection_algorithm=self.__float_to_instance(sol[0], self.__parent.get_feature_selection_algorithms(), self.__feature_selection_algorithm_factory) if self.__parent.get_feature_selection_algorithms() is not None and len(self.__parent.get_feature_selection_algorithms()) > 0 else None,
                feature_transform_algorithm=self.__float_to_instance(sol[1], self.__parent.get_feature_transform_algorithms(), self.__feature_transform_algorithm_factory) if self.__parent.get_feature_transform_algorithms() is not None and len(self.__parent.get_feature_transform_algorithms()) > 0 else None,
                classifier=self.__float_to_instance(sol[2], self.__parent.get_classifiers(), self.__classifier_factory)
            )

            fitness = pipeline.optimize(data.get_x(), data.get_y(), self.__inner_population_size, self.__number_of_inner_evaluations, self.__optimization_algorithm, self.__fitness_name)
            if fitness < self.__current_best_fitness:
                self.__current_best_fitness = fitness
                self.__current_best_pipeline = pipeline

            return fitness
        
        return evaluate