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
	__baggingClassifier = BaggingClassifier()
	_params = dict(
			n_estimators = ParameterDefinition(MinMax(min=5, max=30), np.uint),
			bootstrap = ParameterDefinition([True, False], None),
			bootstrap_features = ParameterDefinition([True, False], None)
		)

	def _set_parameters(self, **kwargs):
		r"""Set the parameters/arguments of the algorithm.

		See Also:
			* :func:`niaaml.classifiers.Classifier._set_parameters`
		"""
		self.__baggingClassifier.set_params(**kwargs)

	def fit(self, x, y, **kwargs):
		r"""Fit BaggingClassifier.

        Arguments:
            x (Iterable[any]): n samples to classify.
			y (numpy.array[int]): n classes of the samples in the x array.

        Returns:
            None
		"""
		self.__baggingClassifier.fit(x, y)

	def predict(self, x, **kwargs):
		r"""Predict class for each sample (row) in x.

        Arguments:
            x (Iterable[any]): n samples to classify.

        Returns:
            numpy.array[int]: n predicted classes.
		"""
		self.__baggingClassifier.predict(x)
