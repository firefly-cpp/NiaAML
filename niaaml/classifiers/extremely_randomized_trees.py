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

	def __init__(self, **kwargs):
		r"""Initialize ExtremelyRandomizedTrees instance.
		"""
		self._params = dict(
			n_estimators = ParameterDefinition(MinMax(min=10, max=200), np.uint),
			criterion = ParameterDefinition(['gini', 'entropy']),
			min_samples_split = ParameterDefinition(MinMax(min=2, max=10), np.uint)
		)
		self.__extra_trees_classifier = ExtraTreesClassifier()

	def set_parameters(self, **kwargs):
		r"""Set the parameters/arguments of the algorithm.
		"""
		self.__extra_trees_classifier.set_params(**kwargs)

	def fit(self, x, y, **kwargs):
		r"""Fit ExtremelyRandomizedTrees.

        Arguments:
            x (Iterable[any]): n samples to classify.
			y (numpy.array[int]): n classes of the samples in the x array.

        Returns:
            None
		"""
		self.__extra_trees_classifier.fit(x, y)

	def predict(self, x, **kwargs):
		r"""Predict class for each sample (row) in x.

        Arguments:
            x (Iterable[any]): n samples to classify.

        Returns:
            numpy.array[int]: n predicted classes.
		"""
		return self.__extra_trees_classifier.predict(x)
