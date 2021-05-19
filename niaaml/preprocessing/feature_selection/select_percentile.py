from niaaml.utilities import ParameterDefinition, MinMax
from niaaml.preprocessing.feature_selection.feature_selection_algorithm import (
    FeatureSelectionAlgorithm,
)
from sklearn.feature_selection import (
    SelectPercentile as SelectPerc,
    chi2,
    f_classif,
    mutual_info_classif,
)
import numpy as np

__all__ = ["SelectPercentile"]


class SelectPercentile(FeatureSelectionAlgorithm):
    r"""Implementation of feature selection using percentile selection of best features according to used score function.

    Date:
        2020

    Author:
        Luka Pečnik

    License:
        MIT

    Documentation:
        https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectPercentile.html

    See Also:
        * :class:`niaaml.preprocessing.feature_selection.feature_selection_algorithm.FeatureSelectionAlgorithm`
    """
    Name = "Select Percentile"

    def __init__(self, **kwargs):
        r"""Initialize SelectPercentile feature selection algorithm."""
        self._params = dict(
            score_func=ParameterDefinition([chi2, f_classif, mutual_info_classif]),
            percentile=ParameterDefinition(MinMax(10, 100), np.uint),
        )
        self.__select_percentile = SelectPerc()

    def set_parameters(self, **kwargs):
        r"""Set the parameters/arguments of the algorithm."""
        self.__select_percentile.set_params(**kwargs)

    def select_features(self, x, y, **kwargs):
        r"""Perform the feature selection process.

        Arguments:
            x (pandas.core.frame.DataFrame): Array of original features.
            y (pandas.core.series.Series) Expected classifier results.

        Returns:
            numpy.ndarray[bool]: Mask of selected features.
        """
        self.__select_percentile.fit(x, y)
        return self.__select_percentile.get_support()

    def to_string(self):
        r"""User friendly representation of the object.

        Returns:
            str: User friendly representation of the object.
        """
        return FeatureSelectionAlgorithm.to_string(self).format(
            name=self.Name,
            args=self._parameters_to_string(self.__select_percentile.get_params()),
        )
