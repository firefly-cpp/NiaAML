from niaaml.utilities import ParameterDefinition, MinMax
from niaaml.preprocessing.feature_selection.feature_selection_algorithm import (
    FeatureSelectionAlgorithm,
)
from sklearn.feature_selection import (
    SelectKBest as SelectKB,
    chi2,
    f_classif,
    mutual_info_classif,
)
import numpy as np

__all__ = ["SelectKBest"]


class SelectKBest(FeatureSelectionAlgorithm):
    r"""Implementation of feature selection using selection of k best features according to used score function.

    Date:
        2020

    Author:
        Luka Pečnik

    License:
        MIT

    Documentation:
        https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html

    See Also:
        * :class:`niaaml.preprocessing.feature_selection.feature_selection_algorithm.FeatureSelectionAlgorithm`
    """
    Name = "Select K Best"

    def __init__(self, **kwargs):
        r"""Initialize SelectKBest feature selection algorithm.

        Notes:
            _params['k'] is initialized to None as it is included in the optimization process later since we cannot determine a proper value range until length of the feature vector becomes known.
        """
        self._params = dict(
            score_func=ParameterDefinition([chi2, f_classif, mutual_info_classif]),
            k=None,
        )
        self.__k = None
        self.__select_k_best = SelectKB()

    def set_parameters(self, **kwargs):
        r"""Set the parameters/arguments of the algorithm."""
        self.__select_k_best.set_params(**kwargs)

    def select_features(self, x, y, **kwargs):
        r"""Perform the feature selection process.

        Arguments:
            x (pandas.core.frame.DataFrame): Array of original features.
            y (pandas.core.series.Series) Expected classifier results.

        Returns:
            numpy.ndarray[bool]: Mask of selected features.
        """
        if self.__k is None:
            self.__k = x.shape[1]
            self._params["k"] = ParameterDefinition(MinMax(1, self.__k), int)
            val = int(np.around(np.random.uniform(1, self.__k)))
            self.__select_k_best.set_params(k=val)

        self.__select_k_best.fit(x, y)
        return self.__select_k_best.get_support()

    def to_string(self):
        r"""User friendly representation of the object.

        Returns:
            str: User friendly representation of the object.
        """
        return FeatureSelectionAlgorithm.to_string(self).format(
            name=self.Name,
            args=self._parameters_to_string(self.__select_k_best.get_params()),
        )
