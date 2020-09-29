from niaaml.classifiers.classifier import Classifier
from niaaml.utilities import MinMax
from niaaml.utilities import ParameterDefinition
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np

__all__ = ['ExtremelyRandomizedTrees']

class ExtremelyRandomizedTrees(Classifier):
	r"""Implementation of extremely randomized trees classifier.
	
	Date:
		2020

	Author
		Luka Pečnik

	License:
		MIT

	See Also:
		* :class:`niaaml.classifiers.Classifier`
	"""
	__extraTreesClassifier = ExtraTreesClassifier()
	_params = dict(
			n_estimators = ParameterDefinition(MinMax(min=10, max=200), np.uint),
			criterion = ParameterDefinition(['gini', 'entropy'], None),
			min_samples_split = ParameterDefinition(MinMax(min=2, max=10), np.uint)
		)

	def _set_parameters(self, **kwargs):
		r"""Set the parameters/arguments of the algorithm.

		See Also:
			* :func:`niaaml.classifiers.Classifier._set_parameters`
		"""
		self.__extraTreesClassifier.set_params(**kwargs)

	def fit(self, x, y, **kwargs):
		r"""Fit ExtraTreesClassifier.

        Arguments:
            x (numpy.ndarray[float]): n samples to classify.
			y (numpy.array[int]): n classes of the samples in the x array.

        Returns:
            None
		"""
		self.__extraTreesClassifier.fit(x, y)

	def predict(self, x, **kwargs):
		r"""Predict class for each sample (row) in x.

        Arguments:
            x (numpy.ndarray[float]): n samples to classify.

        Returns:
            numpy.array[int]: n predicted classes.
		"""
		self.__extraTreesClassifier.predict(x)
