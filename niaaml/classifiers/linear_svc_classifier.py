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

	def __init__(self, **kwargs):
		r"""Initialize LinearSVCClassifier instance.
		"""
		self._params = dict(
			penalty = ParameterDefinition(['l1', 'l2']),
			max_iter = ParameterDefinition(MinMax(min=300, max=2000), np.uint)
		)
		self.__linear_SVC = LinearSVC()

	def set_parameters(self, **kwargs):
		r"""Set the parameters/arguments of the algorithm.
		"""
		self.__linear_SVC.set_params(**kwargs)

	def fit(self, x, y, **kwargs):
		r"""Fit LinearSVCClassifier.

        Arguments:
            x (Iterable[any]): n samples to classify.
			y (numpy.array[int]): n classes of the samples in the x array.

        Returns:
            None
		"""
		self.__linear_SVC.fit(x, y)

	def predict(self, x, **kwargs):
		r"""Predict class for each sample (row) in x.

        Arguments:
            x (Iterable[any]): n samples to classify.

        Returns:
            numpy.array[int]: n predicted classes.
		"""
		return self.__linear_SVC.predict(x)
