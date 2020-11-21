from niaaml.classifiers.classifier import Classifier
from niaaml.utilities import MinMax
from niaaml.utilities import ParameterDefinition
from sklearn.ensemble import AdaBoostClassifier
import numpy as np

__all__ = ['AdaBoost']

class AdaBoost(Classifier):
	r"""Implementation of AdaBoost classifier.
	
	Date:
		2020

	Author
		Luka Pečnik

	License:
		MIT

	See Also:
		* :class:`niaaml.classifiers.Classifier`
	"""
	
	def __init__(self, **kwargs):
		r"""Initialize AdaBoost instance.
		"""
		self._params = dict(
			n_estimators = ParameterDefinition(MinMax(min=10, max=150), np.uint),
			algorithm = ParameterDefinition(['SAMME', 'SAMME.R'])
		)
		self.__ada_boost = AdaBoostClassifier()

	def _set_parameters(self, **kwargs):
		r"""Set the parameters/arguments of the algorithm.

		See Also:
			* :func:`niaaml.classifiers.Classifier._set_parameters`
		"""
		self.__ada_boost.set_params(**kwargs)

	def fit(self, x, y, **kwargs):
		r"""Fit AdaBoost.

        Arguments:
            x (Iterable[any]): n samples to classify.
			y (numpy.array[int]): n classes of the samples in the x array.

        Returns:
            None
		"""
		self.__ada_boost.fit(x, y)

	def predict(self, x, **kwargs):
		r"""Predict class for each sample (row) in x.

        Arguments:
            x (Iterable[any]): n samples to classify.

        Returns:
            numpy.array[int]: n predicted classes.
		"""
		return self.__ada_boost.predict(x)
