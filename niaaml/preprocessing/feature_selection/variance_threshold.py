from niaaml.utilities import ParameterDefinition, MinMax
from niaaml.preprocessing.feature_selection.feature_selection_algorithm import (
    FeatureSelectionAlgorithm,
)
from sklearn.feature_selection import VarianceThreshold as VarThr
import numpy as np

__all__ = ["VarianceThreshold"]


class VarianceThreshold(FeatureSelectionAlgorithm):
    r"""Implementation of feature selection using variance threshold.

    Date:
        2020

    Author:
        Luka Pečnik

    License:
        MIT

    Documentation:
        https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html

    See Also:
        * :class:`niaaml.preprocessing.feature_selection.feature_selection_algorithm.FeatureSelectionAlgorithm`
    """
    Name = "Variance Threshold"

    def __init__(self, **kwargs):
        r"""Initialize VarianceThreshold feature selection algorithm."""
        self._params = dict(threshold=ParameterDefinition(MinMax(0, 0.1), float))
        self.__variance_threshold = VarThr()

    def set_parameters(self, **kwargs):
        r"""Set the parameters/arguments of the algorithm."""
        self.__variance_threshold.set_params(**kwargs)

    def select_features(self, x, y, **kwargs):
        r"""Perform the feature selection process.

        Arguments:
            x (pandas.core.frame.DataFrame): Array of original features.
            y (pandas.core.series.Series) Expected classifier results.

        Returns:
            numpy.ndarray[bool]: Mask of selected features.
        """
        self.__variance_threshold.fit(x)
        return self.__variance_threshold.get_support()

    def to_string(self):
        r"""User friendly representation of the object.

        Returns:
            str: User friendly representation of the object.
        """
        return FeatureSelectionAlgorithm.to_string(self).format(
            name=self.Name,
            args=self._parameters_to_string(self.__variance_threshold.get_params()),
        )
