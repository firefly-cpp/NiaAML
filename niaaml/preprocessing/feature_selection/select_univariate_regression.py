from niaaml.utilities import ParameterDefinition
from niaaml.preprocessing.feature_selection.feature_selection_algorithm import (
    FeatureSelectionAlgorithm,
)
from sklearn.feature_selection import (
    GenericUnivariateSelect as Select,
    r_regression
)

__all__ = ["SelectUnivariateRegression"]


class SelectUnivariateRegression(FeatureSelectionAlgorithm):
    r"""Implementation of feature selection using a generic univariate selection strategy from scikit learn.

    Date:
        2024

    Author:
        Laurenz Farthofer

    License:
        MIT

    Documentation:
        https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.GenericUnivariateSelect.html#sklearn.feature_selection.GenericUnivariateSelect

    See Also:
        * :class:`niaaml.preprocessing.feature_selection.feature_selection_algorithm.FeatureSelectionAlgorithm`
    """
    Name = "Select Univariate Regression"

    def __init__(self, **kwargs):
        r"""Initialize SelectPercentile feature selection algorithm."""
        self._params = dict(
            score_func=ParameterDefinition([r_regression]),
        )
        self.__select = Select()

    def set_parameters(self, **kwargs):
        r"""Set the parameters/arguments of the algorithm."""
        self.__select.set_params(**kwargs)

    def select_features(self, x, y, **kwargs):
        r"""Perform the feature selection process.

        Arguments:
            x (pandas.core.frame.DataFrame): Array of original features.
            y (pandas.core.series.Series) Expected classifier results.

        Returns:
            numpy.ndarray[bool]: Mask of selected features.
        """
        self.__select.fit(x, y)
        return self.__select.get_support()

    def to_string(self):
        r"""User friendly representation of the object.

        Returns:
            str: User friendly representation of the object.
        """
        return FeatureSelectionAlgorithm.to_string(self).format(
            name=self.Name,
            args=self._parameters_to_string(self.__select.get_params()),
        )
