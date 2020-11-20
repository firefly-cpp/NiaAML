from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
    
    def optimize(self, population_size, number_of_evaluations):
        r"""TODO
        """
        # TODO implement optimization process
        try:
            X = self.__data.get_x()

            if self.__feature_selection_algorithm is not None:
                X = self.__feature_selection_algorithm.select_features(self.__data.get_x(), self.__data.get_y())
            
            if self.__feature_transform_algorithm is not None:
                X = self.__feature_transform_algorithm.transform(X)
            
            train_X, test_X, train_y, test_y = train_test_split(
                X, self.__data.get_y(), test_size=0.2)

            self.__classifier.fit(train_X, train_y)
            predictions = self.__classifier.predict(test_X)

            return accuracy_score(test_y, predictions)
        except:
            # infeasible solution as it causes some kind of error
            # return negative infinity as we are looking for maximum accuracy in the optimization process
            return float('-inf')
