from niaaml.classifiers.classifier import Classifier
from niaaml.utilities import MinMax
from niaaml.utilities import ParameterDefinition
from sklearn.ensemble import BaggingClassifier
import numpy as np

__all__ = ['Bagging']

class Bagging(Classifier):
	r"""Implementation of bagging classifier.
	
	Date:
		2020

	Author
		Luka Pečnik

	License:
		MIT

	See Also:
		* :class:`niaaml.classifiers.Classifier`
	"""
	
	_params = dict(
			n_estimators = ParameterDefinition(MinMax(min=10, max=150), np.uint),
			bootstrap = ParameterDefinition([True, False]),
			bootstrap_features = ParameterDefinition([True, False])
		)

	def __init__(self, **kwargs):
		r"""Initialize Bagging instance.
		"""
		self.__bagging_classifier = BaggingClassifier()

	def _set_parameters(self, **kwargs):
		r"""Set the parameters/arguments of the algorithm.

		See Also:
			* :func:`niaaml.classifiers.Classifier._set_parameters`
		"""
		self.__bagging_classifier.set_params(**kwargs)

	def fit(self, x, y, **kwargs):
		r"""Fit Bagging.

        Arguments:
            x (Iterable[any]): n samples to classify.
			y (numpy.array[int]): n classes of the samples in the x array.

        Returns:
            None
		"""
		self.__bagging_classifier.fit(x, y)

	def predict(self, x, **kwargs):
		r"""Predict class for each sample (row) in x.

        Arguments:
            x (Iterable[any]): n samples to classify.

        Returns:
            numpy.array[int]: n predicted classes.
		"""
		return self.__bagging_classifier.predict(x)
