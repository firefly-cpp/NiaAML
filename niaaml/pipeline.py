from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from niaaml.utilities import MinMax, get_bin_index, OptimizationStats
from niaaml.fitness import FitnessFactory
from NiaPy.benchmarks import Benchmark
from NiaPy.algorithms.utility import AlgorithmUtility
from NiaPy.task import StoppingTask
import numpy as np
import copy
import pickle
import os

__all__ = [
    'Pipeline'
]

class Pipeline:
    r"""Classification pipeline defined by optional preprocessing steps and classifier.
    
	Date:
		2020

	Author
		Luka Pečnik

	License:
        MIT

	Attributes:
		__data (DataReader): Instance of any DataReader implementation.
		__feature_selection_algorithm (Optional[FeatureSelectionAlgorithm]): Feature selection algorithm implementation.
		__feature_transform_algorithm (Optional[FeatureTransformAlgorithm]): Feature transform algorithm implementation.
        __classifier (Classifier): Classifier implementation.
        __selected_features_mask (Iterable[bool]): Mask of selected features during the feature selection process.
        __best_stats (OptimizationStats): Statistics of the most successful setup of parameters.
        __niapy_algorithm_utility (AlgorithmUtility): Class used for getting an optimiziation algorithm using its name.
    """

    def __init__(self, **kwargs):
        r"""Initialize task.
        """
        self.__data = None
        self.__feature_selection_algorithm = None
        self.__feature_transform_algorithm = None
        self.__classifier = None
        self.__selected_features_mask = None
        self.__best_stats = None
        self.__niapy_algorithm_utility = AlgorithmUtility()
        self._set_parameters(**kwargs)
    
    def _set_parameters(self, data, feature_selection_algorithm, feature_transform_algorithm, classifier, **kwargs):
        r"""Set the parameters/arguments of the task.

		Arguments:
            data (DataReader): Instance of any DataReader implementation.
            feature_selection_algorithm (Optional[FeatureSelectionAlgorithm]): Feature selection algorithm implementation.
            feature_transform_algorithm (Optional[FeatureTransformAlgorithm]): Feature transform algorithm implementation.
            classifier (Classifier): Classifier implementation.
        """
        self.__data = data
        self.__feature_selection_algorithm = feature_selection_algorithm
        self.__feature_transform_algorithm = feature_transform_algorithm
        self.__classifier = classifier
    
    def get_data(self):
        r"""Get deep copy of the data.

        Returns:
            DataReader: Instance of the DataReader object.
        """
        return copy.deepcopy(self.__data)
    
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
    
    def set_feature_selection_algorithm(self, value):
        r"""Set feature selection algorithm.
        """
        self.__feature_selection_algorithm = value
    
    def set_feature_transform_algorithm(self, value):
        r"""Set feature transform algorithm.
        """
        self.__feature_transform_algorithm = value
    
    def set_classifier(self, value):
        r"""Set classifier.
        """
        self.__classifier = value
    
    def set_selected_features_mask(self, value):
        r"""Set selected features mask.
        """
        self.__selected_features_mask = value
    
    def set_stats(self, value):
        r"""Set stats.
        """
        self.__best_stats = value
    
    def optimize(self, population_size, number_of_evaluations, optimization_algorithm, fitness_function):
        r"""Optimize pipeline's hyperparameters.

        Arguments:
            population_size (uint): Number of individuals in the optimization process.
            number_of_evaluations (uint): Number of maximum evaluations.
            optimization_algorithm (str): Name of the optimization algorithm to use.
            fitness_function (str): Name of the fitness function to use.
        
        Returns:
            float: Best fitness value found in optimization process.
        """

        D = 0
        if self.__feature_selection_algorithm is not None and self.__feature_selection_algorithm.get_params_dict() is not None:
            D += len(self.__feature_selection_algorithm.get_params_dict().keys())
        if self.__feature_transform_algorithm is not None and self.__feature_transform_algorithm.get_params_dict() is not None:
            D += len(self.__feature_transform_algorithm.get_params_dict().keys())
        if self.__classifier.get_params_dict() is not None:
            D += len(self.__classifier.get_params_dict().keys())

        algo = self.__niapy_algorithm_utility.get_algorithm(optimization_algorithm)
        algo.NP = population_size

        task = StoppingTask(
            D=D,
            nFES=number_of_evaluations,
            benchmark=self._PipelineBenchmark(self, population_size, fitness_function)
            )
        best = algo.run(task)
        return best[1]
    
    def run(self, x):
        r"""Runs the pipeline.

        Arguments:
            x (Iterable[any]): n samples to classify.
        
        Returns:
            Iterable[any]: n predicted classes of the samples in the x array.
        """
        x = x[self.__selected_features_mask] if self.__selected_features_mask is not None else x
        
        if self.__feature_transform_algorithm is not None:
            x = self.__feature_transform_algorithm.transform(x)

        return classifier.predict(x)
    
    def export(self, file_name):
        r"""Exports Pipeline object to a file for later use.

        Arguments:
            file_name (str): Output file name.
        """
        pipeline = Pipeline(
            data=None,
            feature_selection_algorithm=self.__feature_selection_algorithm,
            feature_transform_algorithm=self.__feature_transform_algorithm,
            classifier=self.__classifier
        )
        pipeline.set_selected_features_mask(self.__selected_features_mask)
        if len(os.path.splitext(file_name)[1]) == 0 or os.path.splitext(file_name)[1] != '.ppln':
            file_name = file_name + '.ppln'

        with open(file_name, 'wb') as f:
            pickle.dump(pipeline, f)
    
    @staticmethod
    def load(file_name):
        r"""Loads Pipeline object from a file.

        Returns:
            Pipeline: Loaded Pipeline instance.
        """
        with open(file_name, 'rb') as f:
            return pickle.load(f)
    
    def to_string(self):
        r"""User friendly representation of the object.

        Returns:
            str: User friendly representation of the object.
        """
        classifier_string = '\t' + self.__classifier.to_string().replace('\n', '\n\t')
        feature_selection_algorithm_string = '\t' + self.__feature_selection_algorithm.to_string().replace('\n', '\n\t') if self.__feature_selection_algorithm is not None else '\tNone'
        feature_transform_algorithm_string = '\t' + self.__feature_transform_algorithm.to_string().replace('\n', '\n\t') if self.__feature_transform_algorithm is not None else '\tNone'
        stats_string = '\t' + self.__best_stats.to_string().replace('\n', '\n\t') if self.__best_stats is not None else '\tStatistics is not available.'
        features_string = '\t' + str(self.__selected_features_mask) if self.__selected_features_mask is not None else '\tFeature selection result is not available.'
        return 'Classifier:\n{classifier}\n\nFeature selection algorithm:\n{fsa}\n\nFeature transform algorithm:\n{fta}\n\nMask of selected features (True if selected, False if not):\n{feat}\n\nStatistics:\n{stats}'.format(classifier=classifier_string, fsa=feature_selection_algorithm_string, fta=feature_transform_algorithm_string, feat=features_string, stats=stats_string)

    class _PipelineBenchmark(Benchmark):
        r"""NiaPy Benchmark class implementation.

        Attributes:
            __parent (Pipeline): Parent Pipeline instance.
            __population_size (uint): Number of individuals in the hiperparameter optimization process.
            __current_best_fitness (float): Current best fitness of the optimization process.
            __fitness_function (FitnessFunction): Instance of a FitnessFunction object.
        """

        def __init__(self, parent, population_size, fitness_function):
            r"""Initialize pipeline benchmark.

            Arguments:
                parent (Pipeline): Parent instance of Pipeline.
                population_size (uint): Number of individuals in the hiperparameter optimization process.
                fitness_function (str): Name of the fitness function to use.
            """
            self.__parent = parent
            self.__population_size = population_size
            self.__current_best_fitness = float('inf')
            self.__fitness_function = FitnessFactory().get_result(fitness_function)
            Benchmark.__init__(self, 0.0, 1.0)
        
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
                try:
                    data = self.__parent.get_data()
                    feature_selection_algorithm = self.__parent.get_feature_selection_algorithm()
                    feature_transform_algorithm = self.__parent.get_feature_transform_algorithm()
                    classifier = self.__parent.get_classifier()
                    selected_features_mask = None

                    feature_selection_algorithm_params = feature_selection_algorithm.get_params_dict() if feature_selection_algorithm else dict()
                    feature_transform_algorithm_params = feature_transform_algorithm.get_params_dict() if feature_transform_algorithm else dict()
                    classifier_params = classifier.get_params_dict()

                    params_all = [
                        (feature_selection_algorithm_params, feature_selection_algorithm),
                        (feature_transform_algorithm_params, feature_transform_algorithm),
                        (classifier_params, classifier)
                    ]
                    solution_index = 0
                    for i in params_all:
                        args = dict()
                        for key in i[0]:
                            if i[0][key] is not None:
                                if isinstance(i[0][key].value, MinMax):
                                    val = sol[solution_index] * i[0][key].value.max + i[0][key].value.min
                                    if i[0][key].param_type is np.intc or i[0][key].param_type is np.int or i[0][key].param_type is np.uintc or i[0][key].param_type is np.uint:
                                        val = i[0][key].param_type(np.floor(val))
                                        if val >= i[0][key].value.max:
                                            val = i[0][key].value.max - 1
                                    args[key] = val
                                else:
                                    args[key] = i[0][key].value[get_bin_index(sol[solution_index], len(i[0][key].value))]
                            solution_index += 1
                        i[1].set_parameters(**args)

                    x = data.get_x()
                    y = data.get_y()
                    
                    scores = []
                    kf = StratifiedKFold(n_splits=11, random_state=0, shuffle=True)
                    selected_features_mask = None
                    fit_iteration = True
                    for train_index, test_index in kf.split(x, y):
                        x_train, x_test, y_train, y_test = x[train_index], x[test_index], y[train_index], y[test_index]

                        if fit_iteration:
                            if feature_selection_algorithm is None:
                                selected_features_mask = np.ones(x.shape[1], dtype=bool)
                            else:
                                selected_features_mask = feature_selection_algorithm.select_features(x_train, y_train)
                            x = x[selected_features_mask]

                            if feature_transform_algorithm is not None:
                                x_train = x_train[selected_features_mask]
                                feature_transform_algorithm.fit(x_train)
                                feature_transform_algorithm.transform(x)
                            
                            fit_iteration = False
                        else:
                            classifier.fit(x_train, y_train)
                            predictions = classifier.predict(x_test)
                            scores.push(self.__fitness_function.get_fitness(predictions, y_test))
                    
                    fitness = np.mean(scores) * -1

                    if fitness < self.__current_best_fitness:
                        self.__current_best_fitness = fitness
                        self.__parent.set_feature_selection_algorithm(feature_selection_algorithm)
                        self.__parent.set_feature_transform_algorithm(feature_transform_algorithm)
                        self.__parent.set_classifier(classifier)
                        self.__parent.set_selected_features_mask(selected_features_mask)
                        self.__parent.set_stats(OptimizationStats(predictions, y_test))

                    return fitness
                except:
                    # infeasible solution as it causes some kind of error
                    # return infinity as we are looking for maximum accuracy in the optimization process (1 - accuracy since it is a minimization problem)
                    return float('inf')
            
            return evaluate
