__all__ = [
    'Pipeline'
]

class Pipeline:
    r"""Classification pipeline that consists of all components required for the best possible classification.
    
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
        __classificators (Iterable[Classificator]): Array of possible classificators.
    """
    __data = None
    __feature_selection_algorithms = None
    __preprocessing_algorithms = None
    __classificators = None

    def __init__(self, **kwargs):
        r"""Initialize pipeline.
        """
        self._set_parameters(**kwargs)
    
    def _set_parameters(self, data, feature_selection_algorithms, preprocessing_algorithms, classificators, **kwargs):
        r"""Set the parameters/arguments of the pipeline.

		Arguments:
            data (DataReader): Instance of any DataReader implementation.
            feature_selection_algorithms (Iterable[FeatureSelectionAlgorithm]): Array of possible feature selection algorithms.
            preprocessing_algorithms (Iterable[PreprocessingAlgorithm]): Array of possible preprocessing algorithms.
            classificators (Iterable[Classificator]): Array of possible classificators.
        """
        self.__data = data
        self.__feature_selection_algorithms = feature_selection_algorithms
        self.__preprocessing_algorithms = preprocessing_algorithms
        self.__classificators = classificators
