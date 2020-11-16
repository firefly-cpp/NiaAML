__all__ = [
    'Pipeline'
]

class Pipeline:
    r"""Classification pipeline defined by optional preprocessing steps, feature selection algorithm and classifier.
    
	Date:
		2020

	Author
		Luka Pečnik

	License:
        MIT

	Attributes:
		__data (DataReader): Instance of any DataReader implementation.
		__feature_selection_algorithm (FeatureSelectionAlgorithm): Feature selection algorithm implementation.
		__preprocessing_algorithm (PreprocessingAlgorithm): Preprocessing algorithm implementation (optional).
        __classifier (Classifier): Classifier implementation.
    """
    __data = None
    __feature_selection_algorithm = None
    __preprocessing_algorithm = None
    __classifier = None

    def __init__(self, **kwargs):
        r"""Initialize task.
        """
        self._set_parameters(**kwargs)
    
    def _set_parameters(self, data, feature_selection_algorithm, preprocessing_algorithm, classifier, **kwargs):
        r"""Set the parameters/arguments of the task.

		Arguments:
            data (DataReader): Instance of any DataReader implementation.
            feature_selection_algorithm (FeatureSelectionAlgorithm): Feature selection algorithm implementation.
            preprocessing_algorithm (PreprocessingAlgorithm): Preprocessing algorithm implementation (optional).
            classifier (Classifier): Classifier implementation.
        """
        self.__data = data
        self.__feature_selection_algorithm = feature_selection_algorithm
        self.__preprocessing_algorithm = preprocessing_algorithm
        self.__classifier = classifier