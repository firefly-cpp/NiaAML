from niaaml.classifiers.classifier import Classifier
from niaaml.utilities import MinMax
from niaaml.utilities import ParameterDefinition
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
import numpy as np

__all__ = ['QuadraticDiscriminantAnalysis']

class QuadraticDiscriminantAnalysis(Classifier):
    r"""Implementation of quadratic discriminant analysis classifier.
    
    Date:
        2020

    Author:
        Luka Pečnik

    License:
        MIT

    See Also:
        * :class:`niaaml.classifiers.Classifier`
    """
    Name = 'Quadratic Discriminant Analysis'

    def __init__(self, **kwargs):
        r"""Initialize QuadraticDiscriminantAnalysis instance.
        """
        self.__qda = QDA()
        super(QuadraticDiscriminantAnalysis, self).__init__()

    def set_parameters(self, **kwargs):
        r"""Set the parameters/arguments of the algorithm.
        """
        self.__qda.set_params(**kwargs)

    def fit(self, x, y, **kwargs):
        r"""Fit QuadraticDiscriminantAnalysis.

        Arguments:
            x (pandas.core.frame.DataFrame): n samples to classify.
            y (pandas.core.series.Series): n classes of the samples in the x array.

        Returns:
            None
        """
        self.__qda.fit(x, y)

    def predict(self, x, **kwargs):
        r"""Predict class for each sample (row) in x.

        Arguments:
            x (pandas.core.frame.DataFrame): n samples to classify.

        Returns:
            pandas.core.series.Series: n predicted classes.
        """
        return self.__qda.predict(x)

    def to_string(self):
        r"""User friendly representation of the object.

        Returns:
            str: User friendly representation of the object.
        """
        return Classifier.to_string(self).format(name=self.Name, args=self._parameters_to_string(self.__qda.get_params()))