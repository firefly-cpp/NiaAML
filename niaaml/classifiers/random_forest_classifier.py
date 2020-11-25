from niaaml.classifiers.classifier import Classifier
from niaaml.utilities import MinMax
from niaaml.utilities import ParameterDefinition
from sklearn.ensemble import RandomForestClassifier
import numpy as np

__all__ = ['RandomForestClassifier']

class RandomForestClassifier(Classifier):
    r"""Implementation of random forest classifier.
    
    Date:
        2020

    Author
        Luka Pečnik

    License:
        MIT

    See Also:
        * :class:`niaaml.classifiers.Classifier`
    """
    Name = 'Random Forest Classifier'

    def __init__(self, **kwargs):
        r"""Initialize RandomForestClassifier instance.
        """
        self._params = dict(
            n_estimators = ParameterDefinition(MinMax(min=10, max=111), np.uint)
        )
        self.__random_forest_classifier = RandomForestClassifier()

    def set_parameters(self, **kwargs):
        r"""Set the parameters/arguments of the algorithm.
        """
        self.__random_forest_classifier.set_params(**kwargs)

    def fit(self, x, y, **kwargs):
        r"""Fit RandomForestClassifier.

        Arguments:
            x (Iterable[any]): n samples to classify.
            y (Iterable[any]): n classes of the samples in the x array.

        Returns:
            None
        """
        self.__random_forest_classifier.fit(x, y)

    def predict(self, x, **kwargs):
        r"""Predict class for each sample (row) in x.

        Arguments:
            x (Iterable[any]): n samples to classify.

        Returns:
            Iterable[any]: n predicted classes.
        """
        return self.__random_forest_classifier.predict(x)

    def to_string(self):
        r"""User friendly representation of the object.

        Returns:
            str: User friendly representation of the object.
        """
        return Classifier.to_string(self).format(name=self.Name, args=self._parameters_to_string(self.__random_forest_classifier.get_params()))