from niaaml.classifiers.classifier import Classifier
from niaaml.utilities import MinMax
from niaaml.utilities import ParameterDefinition
from sklearn.neural_network import MLPClassifier
import numpy as np

__all__ = ['MultiLayerPerceptron']

class MultiLayerPerceptron(Classifier):
	r"""Implementation of multi-layer perceptron classifier.
	
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
			activation = ParameterDefinition(['identity', 'logistic', 'tanh', 'relu']),
			solver = ParameterDefinition(['lbfgs', 'sgd', 'adam']),
			max_iter = ParameterDefinition(MinMax(min=200, max=500), np.uint),
			learning_rate = ParameterDefinition(['constant', 'invscaling', 'adaptive'])
		)

	def __init__(self, **kwargs):
		r"""Initialize MultiLayerPerceptron instance.
		"""
		self.__multi_layer_perceptron = MLPClassifier()

	def _set_parameters(self, **kwargs):
		r"""Set the parameters/arguments of the algorithm.

		See Also:
			* :func:`niaaml.classifiers.Classifier._set_parameters`
		"""
		self.__multi_layer_perceptron.set_params(**kwargs)

	def fit(self, x, y, **kwargs):
		r"""Fit MultiLayerPerceptron.

        Arguments:
            x (Iterable[any]): n samples to classify.
			y (numpy.array[int]): n classes of the samples in the x array.

        Returns:
            None
		"""
		self.__multi_layer_perceptron.fit(x, y)

	def predict(self, x, **kwargs):
		r"""Predict class for each sample (row) in x.

        Arguments:
            x (Iterable[any]): n samples to classify.

        Returns:
            numpy.array[int]: n predicted classes.
		"""
		return self.__multi_layer_perceptron.predict(x)
