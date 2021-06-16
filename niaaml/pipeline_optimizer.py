import numpy as np
from niaaml.pipeline import Pipeline, _PipelineProblem
from niaaml.classifiers import ClassifierFactory
from niaaml.preprocessing.feature_selection import FeatureSelectionAlgorithmFactory
from niaaml.preprocessing.feature_transform import FeatureTransformAlgorithmFactory
from niapy.task import Task
from niapy.problems import Problem
from niapy.util.factory import get_algorithm
from niaaml.utilities import get_bin_index
from niaaml.preprocessing.encoding.utility import encode_categorical_features
from niaaml.preprocessing.imputation.utility import impute_features
from niaaml.fitness import FitnessFactory
from niaaml.logger import Logger

__all__ = [
    "PipelineOptimizer",
    "_PipelineOptimizerProblem",
    "_PipelineOptimizerProblemV1",
    "_PipelineOptimizerProblemV2",
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
        __categorical_features_encoders (Dict[FeatureEncoder]): Actual instances of FeatureEncoder for all categorical features.
        __imputer (str): Name of the imputer used for features that contain missing values.
        __imputers (Dict[Imputer]): Actual instances of Imputer for all features that contain missing values.
        __logger (Logger): Logger instance.
    """

    def __init__(self, **kwargs):
        r"""Initialize task."""
        self.__data = None
        self.__feature_selection_algorithms = None
        self.__feature_transform_algorithms = None
        self.__classifiers = None
        self.__categorical_features_encoder = None
        self.__categorical_features_encoders = None
        self.__imputer = None
        self.__imputers = None
        self.__logger = None

        self._set_parameters(**kwargs)

    def _set_parameters(
        self,
        data,
        classifiers,
        feature_selection_algorithms=None,
        feature_transform_algorithms=None,
        categorical_features_encoder=None,
        imputer=None,
        log=True,
        log_verbose=False,
        log_output_file=None,
        **kwargs
    ):
        r"""Set the parameters/arguments of the task.

        Arguments:
            data (DataReader): Instance of any DataReader implementation.
            feature_selection_algorithms (Optional[Iterable[str]]): Array of names of possible feature selection algorithms.
            feature_transform_algorithms (Optional[Iterable[str]]): Array of names of possible feature transform algorithms.
            classifiers (Iterable[Classifier]): Array of names of possible classifiers.
            categorical_features_encoder (Optional[str]): Name of the encoder used for categorical features.
            imputer (Optional[str]): Name of the imputer used for features that contain missing values.
            log (Optional(bool)): Log optimization progress.
            log_verbose (Optional(bool)): Log optimization progress, current best pipeline and warnings.
            log_output_file (Optional(str)): Path to the logging output file. Defaults to standard output.
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
        self.__imputer = imputer

        if log is True:
            self.__logger = Logger(verbose=log_verbose, output_file=log_output_file)

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

    def get_logger(self):
        r"""Get logger.

        Returns:
            Logger: Logger instance.
        """
        return self.__logger

    def run(
        self,
        fitness_name,
        pipeline_population_size,
        inner_population_size,
        number_of_pipeline_evaluations,
        number_of_inner_evaluations,
        optimization_algorithm,
        inner_optimization_algorithm=None,
    ):
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

        algo = get_algorithm(optimization_algorithm)
        algo.NP = pipeline_population_size

        features = self.__data.get_x()

        if self.__imputer is not None:
            features, self.__imputers = impute_features(features, self.__imputer)

        if self.__categorical_features_encoder is not None:
            (
                features,
                self.__categorical_features_encoders,
            ) = encode_categorical_features(
                features, self.__categorical_features_encoder
            )

        self.__data.set_x(features)

        problem = _PipelineOptimizerProblemV2(
            self,
            fitness_name,
            inner_population_size,
            number_of_inner_evaluations,
            inner_optimization_algorithm
            if inner_optimization_algorithm is not None
            else optimization_algorithm,
        )
        task = Task(problem=problem, max_evals=number_of_pipeline_evaluations)
        algo.run(task)

        pipeline = problem.get_pipeline()
        if pipeline is not None:
            pipeline.set_categorical_features_encoders(
                self.__categorical_features_encoders
            )
            pipeline.set_imputers(self.__imputers)

        return pipeline

    def run_v1(
        self,
        fitness_name,
        population_size,
        number_of_evaluations,
        optimization_algorithm,
    ):
        r"""Run classification pipeline optimization process according to the original NiaAML paper.

        Reference:
            Fister, Iztok, Milan Zorman, and Dušan Fister. "Continuous Optimizers for Automatic Design and Evaluation of Classification Pipelines." Frontier Applications of Nature Inspired Computation. Springer, Singapore, 2020. 281-301.

        Arguments:
            fitness_name (str): Name of the fitness class to use as a function.
            population_size (uint): Number of individuals in the optimization process.
            number_of_evaluations (uint): Number of maximum evaluations.
            optimization_algorithm (str): Name of the optimization algorithm to use.

        Returns:
            Pipeline: Best pipeline found in the optimization process.
        """

        algo = get_algorithm(optimization_algorithm)
        algo.NP = population_size

        features = self.__data.get_x()

        if self.__imputer is not None:
            features, self.__imputers = impute_features(features, self.__imputer)

        if self.__categorical_features_encoder is not None:
            (
                features,
                self.__categorical_features_encoders,
            ) = encode_categorical_features(
                features, self.__categorical_features_encoder
            )

        self.__data.set_x(features)

        D = 3
        factories = [
            (self.__feature_selection_algorithms, FeatureSelectionAlgorithmFactory()),
            (self.__feature_transform_algorithms, FeatureTransformAlgorithmFactory()),
            (self.__classifiers, ClassifierFactory()),
        ]

        for f in factories:
            m = 0
            if f[0] is not None:
                for e in f[0]:
                    if e is not None:
                        el = f[1].get_result(e)
                        params = el.get_params_dict()
                        if params is not None and len(params) > m:
                            m = len(params)
            D += m

        problem = _PipelineOptimizerProblemV1(D, self, fitness_name)
        task = Task(problem=problem, max_evals=number_of_evaluations)
        algo.run(task)

        pipeline = problem.get_pipeline()
        if pipeline is not None:
            pipeline.set_categorical_features_encoders(
                self.__categorical_features_encoders
            )
            pipeline.set_imputers(self.__imputers)

        return pipeline


class _PipelineOptimizerProblem(Problem):
    r"""NiaPy Problem class base implementation.

    Date:
        2020

    Author:
        Luka Pečnik

    Attributes:
        _parent (PipelineOptimizer): Parent instance of PipelineOptimizer.
        _current_best_fitness (float): Current best fitness of the optimization process.
        _current_best_pipeline (Pipeline): Current best pipeline of the optimization process.
        _fitness_name (str): Name of the fitness class to use as a function.
        _logger (Logger): Instance of the Logger class.
        _evals (int): Number of current evaluation.

        _classifier_factory (ClassifierFactory): Factory for classifiers.
        _feature_transform_algorithm_factory (FeatureTransformAlgorithmFactory): Factory for feature transform algorithms.
        _feature_selection_algorithm_factory (FeatureSelectionAlgorithmFactory): Factory for feature selection algorithms.
    """
    _classifier_factory = ClassifierFactory()
    _feature_transform_algorithm_factory = FeatureTransformAlgorithmFactory()
    _feature_selection_algorithm_factory = FeatureSelectionAlgorithmFactory()

    def __init__(self, dimension, parent, fitness_name):
        r"""Initialize pipeline optimizer problem.

        Arguments:
            dimension (int): Dimension of the problem.
            parent (PipelineOptimizer): Parent instance of PipelineOptimizer.
            fitness_name (str): Name of the fitness class to use as a function.
        """
        self._parent = parent
        self._current_best_fitness = float("inf")
        self._current_best_pipeline = None
        self._fitness_name = fitness_name
        self._evals = 0
        self._logger = self._parent.get_logger()
        super().__init__(dimension, 0.0, 1.0)

    def _float_to_instance(self, value, collection, factory):
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
        return self._current_best_pipeline

    def _evaluate(self, x):
        pass


class _PipelineOptimizerProblemV2(_PipelineOptimizerProblem):
    r"""NiaPy Problem class implementation.

    Date:
        2020

    Author:
        Luka Pečnik

    Attributes:
        __inner_population_size (uint): Number of individuals in the hiperparameter optimization process.
        __number_of_inner_evaluations (uint): Number of maximum inner evaluations.
        __optimization_algorithm (str): Name of the optimization algorithm to use.

    See also:
        * :class:`niaaml.pipeline_optimizer._PipelineOptimizerProblem`
    """

    def __init__(
        self,
        parent,
        fitness_name,
        inner_population_size,
        number_of_inner_evaluations,
        inner_optimization_algorithm,
    ):
        r"""Initialize pipeline optimizer problem.

        Arguments:
            parent (PipelineOptimizer): Parent instance of PipelineOptimizer.
            fitness_name (str): Name of the fitness class to use as a function.
            inner_population_size (uint): Number of individuals in the hiperparameter optimization process.
            number_of_inner_evaluations (uint): Number of maximum inner evaluations.
            inner_optimization_algorithm (str): Name of the optimization algorithm to use.
        """
        super().__init__(3, parent, fitness_name)
        self.__inner_population_size = inner_population_size
        self.__number_of_inner_evaluations = number_of_inner_evaluations
        self.__optimization_algorithm = inner_optimization_algorithm

    def _evaluate(self, x):
        r"""Override fitness function.

        Args:
            x (numpy.ndarray): Solution vector.

        Returns:
            float: Fitness value of `x`.
        """

        self._evals += 1

        data = self._parent.get_data()
        fs_algo = (
            self._float_to_instance(
                x[0],
                self._parent.get_feature_selection_algorithms(),
                self._feature_selection_algorithm_factory,
            )
            if self._parent.get_feature_selection_algorithms() is not None
            and len(self._parent.get_feature_selection_algorithms()) > 0
            else None
        )
        ft_algo = (
            self._float_to_instance(
                x[1],
                self._parent.get_feature_transform_algorithms(),
                self._feature_transform_algorithm_factory,
            )
            if self._parent.get_feature_transform_algorithms() is not None
            and len(self._parent.get_feature_transform_algorithms()) > 0
            else None
        )
        clsf = self._float_to_instance(
            x[2], self._parent.get_classifiers(), self._classifier_factory
        )

        pipeline = Pipeline(
            feature_selection_algorithm=fs_algo,
            feature_transform_algorithm=ft_algo,
            classifier=clsf,
            logger=self._logger,
        )

        if self._logger is not None:
            self._logger.log_progress(
                "Currently optimizing {evals}: {ppln}".format(
                    ppln=pipeline.to_string_slim(), evals=self._evals
                )
            )

        fitness = pipeline.optimize(
            data.get_x(),
            data.get_y(),
            self.__inner_population_size,
            self.__number_of_inner_evaluations,
            self.__optimization_algorithm,
            self._fitness_name,
        )
        if fitness < self._current_best_fitness:
            self._current_best_fitness = fitness
            self._current_best_pipeline = pipeline

            if self._logger is not None:
                self._logger.log_pipeline(
                    "New current best pipeline with fitness {fit}: {ppln}".format(
                        fit=-fitness, ppln=pipeline.to_string_slim()
                    )
                )

        return fitness


class _PipelineOptimizerProblemV1(_PipelineOptimizerProblem):
    r"""NiaPy Problem class implementation.

    Date:
        2020

    Author:
        Luka Pečnik

    See also:
        * :class:`niaaml.pipeline_optimizer._PipelineOptimizerProblem`
    """

    def __init__(self, dimension, parent, fitness_name):
        r"""Initialize pipeline optimizer problem.

        Arguments:
            dimension (int): Dimension of the problem.
            parent (PipelineOptimizer): Parent instance of PipelineOptimizer.
            fitness_name (str): Name of the fitness class to use as a function.
        """
        super().__init__(dimension, parent, fitness_name)
        self.__fitness_function = FitnessFactory().get_result(self._fitness_name)

    def _evaluate(self, x):
        r"""Override fitness function.

        Args:
            x (numpy.ndarray): Solution vector.

        Returns:
            float: Fitness value of `x`.
        """
        self._evals += 1
        if self._logger is not None:
            self._logger.log_progress(
                "Evaluation {evals}".format(evals=self._evals)
            )

        try:
            data = self._parent.get_data()
            fs_algo = (
                self._float_to_instance(
                    x[0],
                    self._parent.get_feature_selection_algorithms(),
                    self._feature_selection_algorithm_factory,
                )
                if self._parent.get_feature_selection_algorithms() is not None
                and len(self._parent.get_feature_selection_algorithms()) > 0
                else None
            )
            ft_algo = (
                self._float_to_instance(
                    x[1],
                    self._parent.get_feature_transform_algorithms(),
                    self._feature_transform_algorithm_factory,
                )
                if self._parent.get_feature_transform_algorithms() is not None
                and len(self._parent.get_feature_transform_algorithms()) > 0
                else None
            )
            clsf = self._float_to_instance(
                x[2], self._parent.get_classifiers(), self._classifier_factory
            )

            pipeline = Pipeline(
                feature_selection_algorithm=fs_algo,
                feature_transform_algorithm=ft_algo,
                classifier=clsf,
                logger=self._logger,
            )

            if self._logger is not None:
                self._logger.log_progress(
                    "Currently optimizing: {ppln}".format(
                        ppln=pipeline.to_string_slim()
                    )
                )

            (
                fitness,
                selected_features_mask,
                stats,
            ) = _PipelineProblem.evaluate_pipeline(
                x[3:],
                fs_algo,
                ft_algo,
                clsf,
                data.get_x(),
                data.get_y(),
                self.__fitness_function,
            )

            if fitness < self._current_best_fitness:
                self._current_best_fitness = fitness
                pipeline.set_feature_selection_algorithm(fs_algo)
                pipeline.set_feature_transform_algorithm(ft_algo)
                pipeline.set_classifier(clsf)
                pipeline.set_stats(stats)
                pipeline.set_selected_features_mask(selected_features_mask)
                self._current_best_pipeline = pipeline

                if self._logger is not None:
                    self._logger.log_pipeline(
                        "New current best pipeline with fitness {fit}: {ppln}".format(
                            fit=-fitness, ppln=pipeline.to_string_slim()
                        )
                    )

            return fitness
        except:
            # infeasible solution as it causes some kind of error
            # return infinity as we are looking for maximum accuracy in the optimization process (1 - accuracy since it is a minimization problem)
            if self._logger is not None:
                self._logger.log_optimization_error(
                    "Optimization failed for: {ppln}".format(
                        ppln=pipeline.to_string_slim()
                    )
                )
            return np.inf
