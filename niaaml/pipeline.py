from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from niaaml.utilities import MinMax, get_bin_index
import numpy as np
import copy

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
    """

    def __init__(self, **kwargs):
        r"""Initialize task.
        """
        self.__data = None
        self.__feature_selection_algorithm = None
        self.__feature_transform_algorithm = None
        self.__classifier = None
        self.__optimization_algorithm = None
        self.__niapy_algorithm_utility = AlgorithmUtility()
        self._set_parameters(**kwargs)
    
    def _set_parameters(self, data, feature_selection_algorithm, feature_transform_algorithm, classifier, optimization_algorithm, **kwargs):
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
        self.__optimization_algorithm = optimization_algorithm
    
    def get_data():
        r"""Get deep copy of the data.

        Returns:
            DataReader: Instance of the DataReader object.
        """
        return copy.deepcopy(self.__data)
    
    def get_feature_selection_algorithm():
        r"""Get deep copy of the data.

        Returns:
            FeatureSelectionAlgorithm: Instance of the FeatureSelectionAlgorithm object.
        """
        return copy.deepcopy(self.__feature_selection_algorithm)
    
    def get_feature_transform_algorithm():
        r"""Get deep copy of the data.

        Returns:
            FeatureTransformAlgorithm: Instance of the FeatureTransformAlgorithm object.
        """
        return copy.deepcopy(self.__feature_transform_algorithm)
    
    def get_classifier():
        r"""Get deep copy of the data.

        Returns:
            Classifier: Instance of the Classifier object.
        """
        return copy.deepcopy(self.__classifier)
    
    def optimize(self, population_size, number_of_evaluations):
        r"""Optimize pipeline's hyperparameters.

        Arguments:
            population_size (uint): Number of individuals in the optimization process.
            number_of_evaluations (uint): Number of maximum evaluations.
        
        Returns:
			Tuple[numpy.ndarray, float]:
				1. Best individuals components found in optimization process.
				2. Best fitness value found in optimization process.
        """

        D = 0
        if self.__feature_selection_algorithm is not None and self.__feature_selection_algorithm.get_params_dict() is not None:
            D += len(self.__feature_selection_algorithm.get_params_dict().keys())
        if self.__feature_transform_algorithm is not None and self.__feature_transform_algorithm.get_params_dict() is not None:
            D += len(self.__feature_transform_algorithm.get_params_dict().keys())
        if self.__classifier.get_params_dict() is not None:
            D += len(self.__classifier.get_params_dict().keys())

        def _initialize_population(task, NP, rnd=np.random, **kwargs):
            r"""NiaPy's InitPopFunc implementation.

            Arguments:
                task (NiaPy.task.Task): Implementation of NiaPy's Task class.
                NP (uint): Population size.
                rnd (any): Random number generator.
            
            Returns:
                Tuple[numpy.ndarray, numpy.ndarray[float]]]:
                    1. New population with shape `{NP, task.D}`.
                    2. New population's function/fitness values.
            """

            pop = np.random.uniform(size=(NP, task.D))
            fpop = np.apply_along_axis(task.eval, 1, pop)
            return pop, fpop

        algo = self.__niapy_algorithm_utility.get_algorithm(self.__optimization_algorithm)
        algo.NP = population_size
        algo.InitPopFunc = _initialize_population

        task = StoppingTask(
            D=D,
            nFES=number_of_pipeline_evaluations,
            benchmark=self._PipelineBenchmark(self, population_size, number_of_evaluations),
            optType=OptimizationType.MAXIMIZATION
            )
        best = algo.run(task)
        return best

    class _PipelineBenchmark(Benchmark):
        r"""NiaPy Benchmark class implementation.

        Attributes:
            __population_size (uint): Number of individuals in the hiperparameter optimization process.
            __number_of_evaluations (uint): Number of maximum evaluations.
        """

        def __init__(self, parent, population_size, number_of_evaluations):
            r"""Initialize pipeline benchmark.

            Arguments:
                parent (Pipeline): Parent instance of Pipeline.
                population_size (uint): Number of individuals in the hiperparameter optimization process.
                number_of_evaluations (uint): Number of maximum inner evaluations.
            """
            self.__parent = parent
            self.__population_size = population_size
            self.__number_of_evaluations = number_of_evaluations
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
                        for key in i:
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
                        i[1].set_parameters(args)

                    X = data.get_x()

                    if feature_selection_algorithm is not None:
                        X = feature_selection_algorithm.select_features(X, data.get_y())
                    
                    if feature_transform_algorithm is not None:
                        X = feature_transform_algorithm.transform(X)
                    
                    train_X, test_X, train_y, test_y = train_test_split(
                        X, data.get_y(), test_size=0.2)

                    classifier.fit(train_X, train_y)
                    predictions = classifier.predict(test_X)

                    return accuracy_score(test_y, predictions)
                except:
                    # infeasible solution as it causes some kind of error
                    # return negative infinity as we are looking for maximum accuracy in the optimization process
                    return float('-inf')
            
            return evaluate
