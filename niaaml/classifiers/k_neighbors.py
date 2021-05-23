from niaaml.classifiers.classifier import Classifier
from niaaml.utilities import ParameterDefinition
from sklearn.neighbors import KNeighborsClassifier as KNC

import warnings
from sklearn.exceptions import (
    ChangedBehaviorWarning,
    ConvergenceWarning,
    DataConversionWarning,
    DataDimensionalityWarning,
    EfficiencyWarning,
    FitFailedWarning,
    NonBLASDotWarning,
    UndefinedMetricWarning,
)

__all__ = ["KNeighbors"]


class KNeighbors(Classifier):
    r"""Implementation of k neighbors classifier.

    Date:
        2020

    Author:
        Luka Pečnik

    License:
        MIT

    Reference:
        “Neighbourhood Components Analysis”, J. Goldberger, S. Roweis, G. Hinton, R. Salakhutdinov, Advances in Neural Information Processing Systems, Vol. 17, May 2005, pp. 513-520.

    Documentation:
        https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

    See Also:
        * :class:`niaaml.classifiers.Classifier`
    """
    Name = "K Neighbors Classifier"

    def __init__(self, **kwargs):
        r"""Initialize KNeighbors instance."""
        warnings.filterwarnings(action="ignore", category=ChangedBehaviorWarning)
        warnings.filterwarnings(action="ignore", category=ConvergenceWarning)
        warnings.filterwarnings(action="ignore", category=DataConversionWarning)
        warnings.filterwarnings(action="ignore", category=DataDimensionalityWarning)
        warnings.filterwarnings(action="ignore", category=EfficiencyWarning)
        warnings.filterwarnings(action="ignore", category=FitFailedWarning)
        warnings.filterwarnings(action="ignore", category=NonBLASDotWarning)
        warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)

        self._params = dict(
            weights=ParameterDefinition(["uniform", "distance"]),
            algorithm=ParameterDefinition(["auto", "ball_tree", "kd_tree", "brute"]),
        )
        self.__kn_classifier = KNC()

    def set_parameters(self, **kwargs):
        r"""Set the parameters/arguments of the algorithm."""
        self.__kn_classifier.set_params(**kwargs)

    def fit(self, x, y, **kwargs):
        r"""Fit KNeighbors.

        Arguments:
            x (pandas.core.frame.DataFrame): n samples to classify.
            y (pandas.core.series.Series): n classes of the samples in the x array.

        Returns:
            None
        """
        self.__kn_classifier.fit(x, y)

    def predict(self, x, **kwargs):
        r"""Predict class for each sample (row) in x.

        Arguments:
            x (pandas.core.frame.DataFrame): n samples to classify.

        Returns:
            pandas.core.series.Series: n predicted classes.
        """
        return self.__kn_classifier.predict(x)

    def to_string(self):
        r"""User friendly representation of the object.

        Returns:
            str: User friendly representation of the object.
        """
        return Classifier.to_string(self).format(
            name=self.Name,
            args=self._parameters_to_string(self.__kn_classifier.get_params()),
        )
