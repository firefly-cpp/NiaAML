from niaaml.classifiers.classifier import Classifier
from niaaml.utilities import MinMax
from niaaml.utilities import ParameterDefinition
from sklearn.ensemble import RandomForestClassifier
import numpy as np

__all__ = ['RandomForestClassifier']

class RandomForestClassifier(Classifier):
	r"""Implementation of random forest classifier.
	
	Date:
		2020

	Author
		Luka Pečnik

	License:
		MIT

	See Also:
		* :class:`niaaml.classifiers.Classifier`
	"""
	__randomForestClassifier = RandomForestClassifier()
	_params = dict(
			n_estimators = ParameterDefinition(MinMax(min=10, max=150), np.uint)
		)

	def _set_parameters(self, **kwargs):
		r"""Set the parameters/arguments of the algorithm.

		See Also:
			* :func:`niaaml.classifiers.Classifier._set_parameters`
		"""
		self.__randomForestClassifier.set_params(**kwargs)

	def fit(self, x, y, **kwargs):
		r"""Fit RandomForestClassifier.

        Arguments:
            x (Iterable[any]): n samples to classify.
			y (numpy.array[int]): n classes of the samples in the x array.

        Returns:
            None
		"""
		self.__randomForestClassifier.fit(x, y)

	def predict(self, x, **kwargs):
		r"""Predict class for each sample (row) in x.

        Arguments:
            x (Iterable[any]): n samples to classify.

        Returns:
            numpy.array[int]: n predicted classes.
		"""
		self.__randomForestClassifier.predict(x)
