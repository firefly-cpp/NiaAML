from sklearn.model_selection import train_test_split
from niaaml.utilities import MinMax, get_bin_index, OptimizationStats
from niaaml.fitness import FitnessFactory
from niapy.problems import Problem
from niapy.util.factory import get_algorithm
from niapy.task import Task
import pandas as pd
import numpy as np
import copy
import pickle
import os

__all__ = ["Pipeline", "_PipelineProblem"]


class Pipeline:
    r"""Classification pipeline defined by optional preprocessing steps and classifier.

    Date:
        2020

    Author:
        Luka Pečnik

    License:
        MIT

    Attributes:
        __feature_selection_algorithm (Optional[FeatureSelectionAlgorithm]): Feature selection algorithm implementation.
        __feature_transform_algorithm (Optional[FeatureTransformAlgorithm]): Feature transform algorithm implementation.
        __classifier (Classifier): Classifier implementation.
        __selected_features_mask (Iterable[bool]): Mask of selected features during the feature selection process.
        __best_stats (OptimizationStats): Statistics of the most successful setup of parameters.
        __categorical_features_encoders (Dict[FeatureEncoder]): Instances of FeatureEncoder for all categorical features.
        __imputers (Dict[Imputer]): Dictionary of instances of Imputer for all columns that contained missing values during optimization process.
        __logger (Logger): Logger instance.
    """

    def __init__(self, **kwargs):
        r"""Initialize task."""
        self.__feature_selection_algorithm = None
        self.__feature_transform_algorithm = None
        self.__classifier = None
        self.__selected_features_mask = None
        self.__best_stats = None
        self.__categorical_features_encoders = None
        self.__imputers = None
        self.__logger = None
        self._set_parameters(**kwargs)

    def _set_parameters(
        self,
        classifier,
        feature_selection_algorithm=None,
        feature_transform_algorithm=None,
        categorical_features_encoders=None,
        imputers=None,
        logger=None,
        **_kwargs
    ):
        r"""Set the parameters/arguments of the task.

        Arguments:
            feature_selection_algorithm (Optional[FeatureSelectionAlgorithm]): Feature selection algorithm implementation.
            feature_transform_algorithm (Optional[FeatureTransformAlgorithm]): Feature transform algorithm implementation.
            classifier (Classifier): Classifier implementation.
            categorical_features_encoders (Dict[FeatureEncoders]): Actual instances of FeatureEncoder for all categorical features.
            imputers (Dict[Imputer]): Instances of Imputer for all features that contained missing values during optimization process.
            logger (Logger): Instance of the Logger class.
        """
        self.__feature_selection_algorithm = feature_selection_algorithm
        self.__feature_transform_algorithm = feature_transform_algorithm
        self.__classifier = classifier
        self.__categorical_features_encoders = categorical_features_encoders
        self.__imputers = imputers
        self.__logger = logger

    def get_feature_selection_algorithm(self):
        r"""Get deep copy of the feature selection algorithm.

        Returns:
            FeatureSelectionAlgorithm: Instance of the FeatureSelectionAlgorithm object.
        """
        return copy.deepcopy(self.__feature_selection_algorithm)

    def get_feature_transform_algorithm(self):
        r"""Get deep copy of the feature transform algorithm.

        Returns:
            FeatureTransformAlgorithm: Instance of the FeatureTransformAlgorithm object.
        """
        return copy.deepcopy(self.__feature_transform_algorithm)

    def get_classifier(self):
        r"""Get deep copy of the classifier.

        Returns:
            Classifier: Instance of the Classifier object.
        """
        return copy.deepcopy(self.__classifier)

    def get_logger(self):
        r"""Get logger.

        Returns:
            Logger: Instance of the Logger object.
        """
        return self.__logger

    def get_stats(self):
        r"""Get optimization statistics.

        Returns:
            OptimizationStats: Instance of the OptimizationStats object.
        """
        return self.__best_stats

    def set_feature_selection_algorithm(self, value):
        r"""Set feature selection algorithm."""
        self.__feature_selection_algorithm = value

    def set_feature_transform_algorithm(self, value):
        r"""Set feature transform algorithm."""
        self.__feature_transform_algorithm = value

    def set_classifier(self, value):
        r"""Set classifier."""
        self.__classifier = value

    def set_selected_features_mask(self, value):
        r"""Set selected features mask."""
        self.__selected_features_mask = value

    def set_stats(self, value):
        r"""Set stats."""
        self.__best_stats = value

    def set_categorical_features_encoders(self, value):
        r"""Set categorical features' encoders."""
        self.__categorical_features_encoders = value

    def set_imputers(self, value):
        r"""Set imputers."""
        self.__imputers = value

    def optimize(
        self,
        x,
        y,
        population_size,
        number_of_evaluations,
        optimization_algorithm,
        fitness_function,
    ):
        r"""Optimize pipeline's hyperparameters.

        Arguments:
            x (pandas.core.frame.DataFrame): n samples to classify.
            y (pandas.core.series.Series): n classes of the samples in the x array.
            population_size (uint): Number of individuals in the optimization process.
            number_of_evaluations (uint): Number of maximum evaluations.
            optimization_algorithm (str): Name of the optimization algorithm to use.
            fitness_function (str): Name of the fitness function to use.

        Returns:
            float: Best fitness value found in optimization process.
        """

        if self.__imputers is not None:
            for key in self.__imputers:
                x.loc[:, key] = self.__imputers[key].transform(x[[key]])

        if self.__categorical_features_encoders is not None:
            to_drop = []
            enc_features = pd.DataFrame()
            cols = [
                col for col in x.columns if not pd.api.types.is_numeric_dtype(x[col])
            ]
            for c in cols:
                self.__categorical_features_encoders[c].fit(x[[c]])
                tr = self.__categorical_features_encoders[c].transform(x[[c]])
                to_drop.append(c)
                enc_features = pd.concat([enc_features, tr], axis=1)
            x = x.drop(to_drop, axis=1)
            x = pd.concat([x, enc_features], axis=1)

        dimension = 0
        if self.__feature_selection_algorithm is not None:
            dimension += len(self.__feature_selection_algorithm.get_params_dict().keys())
        if self.__feature_transform_algorithm is not None:
            dimension += len(self.__feature_transform_algorithm.get_params_dict().keys())

        dimension += len(self.__classifier.get_params_dict().keys())

        algo = get_algorithm(optimization_algorithm)
        algo.NP = population_size

        task = Task(
            problem=_PipelineProblem(dimension, x, y, self, population_size, fitness_function),
            max_evals=number_of_evaluations,
        )
        best = algo.run(task)
        return best[1]

    def run(self, x):
        r"""Runs the pipeline.

        Arguments:
            x (pandas.core.frame.DataFrame): n samples to classify.

        Returns:
            pandas.core.series.Series: n predicted classes of the samples in the x array.
        """
        if self.__imputers is not None:
            for key in self.__imputers:
                x.loc[:, key] = self.__imputers[key].transform(x[[key]])

        if self.__categorical_features_encoders is not None:
            to_drop = []
            enc_features = pd.DataFrame()
            cols = [
                col for col in x.columns if not pd.api.types.is_numeric_dtype(x[col])
            ]
            for c in cols:
                tr = self.__categorical_features_encoders[c].transform(x[[c]])
                to_drop.append(c)
                enc_features = pd.concat([enc_features, tr], axis=1)
            x = x.drop(to_drop, axis=1)
            x = pd.concat([x, enc_features], axis=1)

        x = (
            x.loc[:, self.__selected_features_mask]
            if self.__selected_features_mask is not None
            else x
        )

        if self.__feature_transform_algorithm is not None:
            x = self.__feature_transform_algorithm.transform(x)

        return self.__classifier.predict(x)

    def export(self, file_name):
        r"""Exports Pipeline object to a file for later use. Extension is added if not present.

        Arguments:
            file_name (str): Output file name.
        """
        pipeline = Pipeline(
            feature_selection_algorithm=self.__feature_selection_algorithm,
            feature_transform_algorithm=self.__feature_transform_algorithm,
            classifier=self.__classifier,
            categorical_features_encoders=self.__categorical_features_encoders,
        )
        pipeline.set_selected_features_mask(self.__selected_features_mask)
        pipeline.set_stats(self.__best_stats)
        if (
            len(os.path.splitext(file_name)[1]) == 0
            or os.path.splitext(file_name)[1] != ".ppln"
        ):
            file_name = file_name + ".ppln"

        with open(file_name, "wb") as f:
            pickle.dump(pipeline, f)

    def export_text(self, file_name):
        r"""Exports Pipeline object to a user-friendly text file. Extension is added if not present.

        Arguments:
            file_name (str): Output file name.
        """
        pipeline = Pipeline(
            feature_selection_algorithm=self.__feature_selection_algorithm,
            feature_transform_algorithm=self.__feature_transform_algorithm,
            classifier=self.__classifier,
            categorical_features_encoders=self.__categorical_features_encoders,
        )
        pipeline.set_selected_features_mask(self.__selected_features_mask)
        pipeline.set_stats(self.__best_stats)
        if (
            len(os.path.splitext(file_name)[1]) == 0
            or os.path.splitext(file_name)[1] != ".txt"
        ):
            file_name = file_name + ".txt"

        with open(file_name, "w") as f:
            f.write(pipeline.to_string())

    @staticmethod
    def load(file_name):
        r"""Loads Pipeline object from a file.

        Returns:
            Pipeline: Loaded Pipeline instance.
        """
        with open(file_name, "rb") as f:
            return pickle.load(f)

    def to_string(self):
        r"""User friendly representation of the object.

        Returns:
            str: User friendly representation of the object.
        """
        classifier_string = "\t" + self.__classifier.to_string().replace("\n", "\n\t")
        feature_selection_algorithm_string = (
            "\t" + self.__feature_selection_algorithm.to_string().replace("\n", "\n\t")
            if self.__feature_selection_algorithm is not None
            else "\tNone"
        )
        feature_transform_algorithm_string = (
            "\t" + self.__feature_transform_algorithm.to_string().replace("\n", "\n\t")
            if self.__feature_transform_algorithm is not None
            else "\tNone"
        )
        stats_string = (
            "\t" + self.__best_stats.to_string().replace("\n", "\n\t")
            if self.__best_stats is not None
            else "\tStatistics is not available."
        )
        features_string = (
            "\t" + str(self.__selected_features_mask)
            if self.__selected_features_mask is not None
            else "\tFeature selection result is not available."
        )

        imputers_string = ""
        if self.__imputers is not None:
            imputers_string += "Missing features' imputers (feature's name or index: imputer's name):\n"
            for key in self.__imputers:
                imputers_string += (
                    "\t* " + str(key) + ": " + self.__imputers[key].to_string() + "\n"
                )
            imputers_string += "\n"

        encoders_string = ""
        if self.__categorical_features_encoders is not None:
            encoders_string += "Categorical features' encoders (feature's name or index: encoder's name):\n"
            for key in self.__categorical_features_encoders:
                encoders_string += (
                    "\t* "
                    + str(key)
                    + ": "
                    + self.__categorical_features_encoders[key].to_string()
                    + "\n"
                )
            encoders_string += "\n"

        return "Classifier:\n{classifier}\n\nFeature selection algorithm:\n{fsa}\n\nFeature transform algorithm:\n{fta}\n\nMask of selected features (True if selected, False if not):\n{feat}\n\n{imp}{enc}Statistics:\n{stats}".format(
            classifier=classifier_string,
            fsa=feature_selection_algorithm_string,
            fta=feature_transform_algorithm_string,
            imp=imputers_string,
            enc=encoders_string,
            feat=features_string,
            stats=stats_string,
        )

    def to_string_slim(self):
        r"""Slim user friendly representation of the object.

        Returns:
            str: Slim user friendly representation of the object.
        """

        return "classifier - {classifier}, feature selection algorithm - {fsa}, feature transform algorithm - {fta}".format(
            classifier=self.__classifier.Name,
            fsa=self.__feature_selection_algorithm.Name
            if self.__feature_selection_algorithm is not None
            else None,
            fta=self.__feature_transform_algorithm.Name
            if self.__feature_transform_algorithm is not None
            else None,
        )


class _PipelineProblem(Problem):
    r"""NiaPy Problem class implementation.

    Date:
        2020

    Author:
        Luka Pečnik

    License:
        MIT

    Attributes:
        __parent (Pipeline): Parent Pipeline instance.
        __x (pandas.core.frame.DataFrame): n samples to classify.
        __y (pandas.core.series.Series): n classes of the samples in the __x array.
        __population_size (uint): Number of individuals in the hiperparameter optimization process.
        __current_best_fitness (float): Current best fitness of the optimization process.
        __fitness_function (FitnessFunction): Instance of a FitnessFunction object.
        __logger (Logger): Instance of the Logger class.
        __evals (int): Number of current evaluation.
    """

    def __init__(self, dimension, x, y, parent, population_size, fitness_function):
        r"""Initialize pipeline problem.

        Arguments:
            dimension (int): Dimension of the problem.
            parent (Pipeline): Parent instance of Pipeline.
            population_size (uint): Number of individuals in the hiperparameter optimization process.
            fitness_function (str): Name of the fitness function to use.
        """
        self.__parent = parent
        self.__x = x
        self.__y = y
        self.__population_size = population_size
        self.__current_best_fitness = np.inf
        self.__fitness_function = FitnessFactory().get_result(fitness_function)
        self.__evals = 0
        self.__logger = self.__parent.get_logger()
        super().__init__(dimension, 0.0, 1.0)

    @staticmethod
    def evaluate_pipeline(
        solution_vector,
        feature_selection_algorithm,
        feature_transform_algorithm,
        classifier,
        x,
        y,
        fitness_function,
    ):
        """Evaluate pipeline setup.

        Arguments:
            solution_vector (numpy.ndarray[float]): Individual of population/ possible solution to map hyperparameters from.
            feature_selection_algorithm (Optional[FeatureSelectionAlgorithm]): Feature selection algorithm instance.
            feature_transform_algorithm (Optional[FeatureTransformAlgorithm]): Feature transform algorithm instance.
            classifier (Classifier): Classifier instance.
            x (pandas.core.frame.DataFrame): n samples to classify.
            y (pandas.core.series.Series): n classes of the samples in the x array.
            fitness_function (FitnessFunction): Fitness function instance.

        Returns:
            Tuple[float, numpy.array[bool], OptimizationStats]:
                1. Fitness.
                2. Feature selection mask.
                3. Optimization statistics.
        """
        feature_selection_algorithm_params = (
            feature_selection_algorithm.get_params_dict()
            if feature_selection_algorithm
            else dict()
        )
        feature_transform_algorithm_params = (
            feature_transform_algorithm.get_params_dict()
            if feature_transform_algorithm
            else dict()
        )
        classifier_params = classifier.get_params_dict()

        params_all = [
            (feature_selection_algorithm_params, feature_selection_algorithm),
            (feature_transform_algorithm_params, feature_transform_algorithm),
            (classifier_params, classifier),
        ]
        solution_index = 0

        for i in params_all:
            args = dict()
            for key in i[0]:
                if i[0][key] is not None:
                    if isinstance(i[0][key].value, MinMax):
                        val = (
                            solution_vector[solution_index] * i[0][key].value.max
                            + i[0][key].value.min
                        )
                        if (
                            i[0][key].param_type is np.intc
                            or i[0][key].param_type is int
                            or i[0][key].param_type is np.uintc
                            or i[0][key].param_type is np.uint
                        ):
                            val = i[0][key].param_type(np.floor(val))
                            if val >= i[0][key].value.max:
                                val = i[0][key].value.max - 1
                        args[key] = val
                    else:
                        args[key] = i[0][key].value[
                            get_bin_index(
                                solution_vector[solution_index], len(i[0][key].value)
                            )
                        ]
                solution_index += 1
            if i[1] is not None:
                i[1].set_parameters(**args)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        if feature_selection_algorithm is None:
            selected_features_mask = np.ones(x.shape[1], dtype=bool)
        else:
            selected_features_mask = feature_selection_algorithm.select_features(
                x_train, y_train
            )

        x_train = x_train.loc[:, selected_features_mask]
        x_test = x_test.loc[:, selected_features_mask]

        if feature_transform_algorithm is not None:
            feature_transform_algorithm.fit(x_train)
            x_train = feature_transform_algorithm.transform(x_train)
            x_test = feature_transform_algorithm.transform(x_test)

        classifier.fit(x_train, y_train)
        predictions = classifier.predict(x_test)
        return (
            fitness_function.get_fitness(predictions, y_test) * -1,
            selected_features_mask,
            OptimizationStats(predictions, y_test),
        )

    def _evaluate(self, x):
        r"""Override fitness function.

        Args:
            x (numpy.ndarray): Solution:

        Returns:
            float: Fitness value of `x`.

        """
        self.__evals += 1
        if self.__logger is not None:
            self.__logger.log_progress(
                "Evaluation {evals}".format(evals=self.__evals)
            )

        try:
            feature_selection_algorithm = (
                self.__parent.get_feature_selection_algorithm()
            )
            feature_transform_algorithm = (
                self.__parent.get_feature_transform_algorithm()
            )
            classifier = self.__parent.get_classifier()

            (
                fitness,
                selected_features_mask,
                stats,
            ) = _PipelineProblem.evaluate_pipeline(
                x,
                feature_selection_algorithm,
                feature_transform_algorithm,
                classifier,
                self.__x,
                self.__y,
                self.__fitness_function,
            )

            if fitness < self.__current_best_fitness:
                self.__current_best_fitness = fitness
                self.__parent.set_feature_selection_algorithm(
                    feature_selection_algorithm
                )
                self.__parent.set_feature_transform_algorithm(
                    feature_transform_algorithm
                )
                self.__parent.set_classifier(classifier)
                self.__parent.set_selected_features_mask(selected_features_mask)
                self.__parent.set_stats(stats)

            return fitness
        except Exception:
            # infeasible solution as it causes some kind of error
            # return infinity as we are looking for maximum accuracy in the optimization process (1 - accuracy since it is a minimization problem)
            if self.__logger is not None:
                self.__logger.log_optimization_error(
                    "Optimization failed for: {ppln}".format(
                        ppln=self.__parent.to_string_slim()
                    )
                )
            return np.inf
