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

    Author:
        Luka Pečnik

    License:
        MIT

    See Also:
        * :class:`niaaml.classifiers.Classifier`
    """
    Name = 'Bagging'

    def __init__(self, **kwargs):
        r"""Initialize Bagging instance.
        """
        self._params = dict(
            n_estimators = ParameterDefinition(MinMax(min=10, max=111), np.uint),
            bootstrap = ParameterDefinition([True, False]),
            bootstrap_features = ParameterDefinition([True, False])
        )
        self.__bagging_classifier = BaggingClassifier()

    def set_parameters(self, **kwargs):
        r"""Set the parameters/arguments of the algorithm.
        """
        self.__bagging_classifier.set_params(**kwargs)

    def fit(self, x, y, **kwargs):
        r"""Fit Bagging.

        Arguments:
            x (pandas.core.frame.DataFrame): n samples to classify.
            y (pandas.core.series.Series): n classes of the samples in the x array.

        Returns:
            None
        """
        self.__bagging_classifier.fit(x, y)

    def predict(self, x, **kwargs):
        r"""Predict class for each sample (row) in x.

        Arguments:
            x (pandas.core.frame.DataFrame): n samples to classify.

        Returns:
            pandas.core.series.Series: n predicted classes.
        """
        return self.__bagging_classifier.predict(x)

    def to_string(self):
        r"""User friendly representation of the object.

        Returns:
            str: User friendly representation of the object.
        """
        return Classifier.to_string(self).format(name=self.Name, args=self._parameters_to_string(self.__bagging_classifier.get_params()))