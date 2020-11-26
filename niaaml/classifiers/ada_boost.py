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
    Name = 'AdaBoost'

    def __init__(self, **kwargs):
        r"""Initialize AdaBoost instance.
        """
        self._params = dict(
            n_estimators = ParameterDefinition(MinMax(min=10, max=111), np.uint),
            algorithm = ParameterDefinition(['SAMME', 'SAMME.R'])
        )
        self.__ada_boost = AdaBoostClassifier()

    def set_parameters(self, **kwargs):
        r"""Set the parameters/arguments of the algorithm.
        """
        self.__ada_boost.set_params(**kwargs)

    def fit(self, x, y, **kwargs):
        r"""Fit AdaBoost.

        Arguments:
            x (numpy.ndarray[float]): n samples to classify.
            y (Iterable[any]): n classes of the samples in the x array.
        """
        self.__ada_boost.fit(x, y)

    def predict(self, x, **kwargs):
        r"""Predict class for each sample (row) in x.

        Arguments:
            x (numpy.ndarray[float]): n samples to classify.

        Returns:
            Iterable[any]: n predicted classes.
        """
        return self.__ada_boost.predict(x)

    def to_string(self):
        r"""User friendly representation of the object.

        Returns:
            str: User friendly representation of the object.
        """
        return Classifier.to_string(self).format(name=self.Name, args=self._parameters_to_string(self.__ada_boost.get_params()))