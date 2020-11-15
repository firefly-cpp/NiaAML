import numpy as np

__all__ = [
    'Task'
]

class Task:
    r"""Task that finds the best classification pipeline according to the given input.
    
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
        __classifiers (Iterable[Classificator]): Array of possible classifiers.
    """
    __data = None
    __feature_selection_algorithms = None
    __preprocessing_algorithms = None
    __classifiers = None

    def __init__(self, **kwargs):
        r"""Initialize task.
        """
        self._set_parameters(**kwargs)
    
    def _set_parameters(self, data, feature_selection_algorithms, preprocessing_algorithms, classifiers, **kwargs):
        r"""Set the parameters/arguments of the task.

		Arguments:
            data (DataReader): Instance of any DataReader implementation.
            feature_selection_algorithms (Iterable[FeatureSelectionAlgorithm]): Array of possible feature selection algorithms.
            preprocessing_algorithms (Iterable[PreprocessingAlgorithm]): Array of possible preprocessing algorithms.
            classifiers (Iterable[Classificator]): Array of possible classifiers.
        """
        self.__data = data
        self.__feature_selection_algorithms = feature_selection_algorithms
        self.__preprocessing_algorithms = preprocessing_algorithms
        self.__classifiers = classifiers

    def __initialize_population(self, pop_size):
        r"""Initialize population of pipelines to find the best setup.

        Arguments:
            pop_size (int): Number of individuals.

        Returns:
            TODO
        """
        return np.random.uniform(size=(pop_size, 3))