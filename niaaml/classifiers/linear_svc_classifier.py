from niaaml.classifiers.classifier import Classifier
from niaaml.utilities import MinMax
from niaaml.utilities import ParameterDefinition
from sklearn.svm import LinearSVC
import numpy as np

__all__ = ['LinearSVCClassifier']

class LinearSVCClassifier(Classifier):
	r"""Implementation of linear support vector classification.
	
	Date:
		2020

	Author
		Luka Pečnik

	License:
		MIT

	See Also:
		* :class:`niaaml.classifiers.Classifier`
	"""
	__linearSVC = LinearSVC()
	_params = dict(
			penalty = ParameterDefinition(['l1', 'l2'], None),
			max_iter = ParameterDefinition(MinMax(min=300, max=2000), np.uint)
		)

	def _set_parameters(self, **kwargs):
		r"""Set the parameters/arguments of the algorithm.

		See Also:
			* :func:`niaaml.classifiers.Classifier._set_parameters`
		"""
		self.__linearSVC.set_params(**kwargs)

	def fit(self, x, y, **kwargs):
		r"""Fit LinearSVC.

        Arguments:
            x (Iterable[any]): n samples to classify.
			y (numpy.array[int]): n classes of the samples in the x array.

        Returns:
            None
		"""
		self.__linearSVC.fit(x, y)

	def predict(self, x, **kwargs):
		r"""Predict class for each sample (row) in x.

        Arguments:
            x (Iterable[any]): n samples to classify.

        Returns:
            numpy.array[int]: n predicted classes.
		"""
		self.__linearSVC.predict(x)
