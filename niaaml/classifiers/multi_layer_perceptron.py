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
    Name = 'Multi Layer Perceptron'

    def __init__(self, **kwargs):
        r"""Initialize MultiLayerPerceptron instance.
        """
        self._params = dict(
            activation = ParameterDefinition(['identity', 'logistic', 'tanh', 'relu']),
            solver = ParameterDefinition(['lbfgs', 'sgd', 'adam']),
            learning_rate = ParameterDefinition(['constant', 'invscaling', 'adaptive'])
        )
        self.__multi_layer_perceptron = MLPClassifier()

    def set_parameters(self, **kwargs):
        r"""Set the parameters/arguments of the algorithm.
        """
        self.__multi_layer_perceptron.set_params(**kwargs)

    def fit(self, x, y, **kwargs):
        r"""Fit MultiLayerPerceptron.

        Arguments:
            x (numpy.ndarray[float]): n samples to classify.
            y (Iterable[any]): n classes of the samples in the x array.

        Returns:
            None
        """
        self.__multi_layer_perceptron.fit(x, y)

    def predict(self, x, **kwargs):
        r"""Predict class for each sample (row) in x.

        Arguments:
            x (numpy.ndarray[float]): n samples to classify.

        Returns:
            Iterable[any]: n predicted classes.
        """
        return self.__multi_layer_perceptron.predict(x)

    def to_string(self):
        r"""User friendly representation of the object.

        Returns:
            str: User friendly representation of the object.
        """
        return Classifier.to_string(self).format(name=self.Name, args=self._parameters_to_string(self.__multi_layer_perceptron.get_params()))