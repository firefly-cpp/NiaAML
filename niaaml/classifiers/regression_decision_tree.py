from niaaml.classifiers.classifier import Classifier
from niaaml.utilities import ParameterDefinition
from sklearn.tree import DecisionTreeRegressor as DTR

import warnings
from sklearn.exceptions import (
    ConvergenceWarning,
    DataConversionWarning,
    DataDimensionalityWarning,
    EfficiencyWarning,
    FitFailedWarning,
    UndefinedMetricWarning,
)

__all__ = ["DecisionTreeRegression"]


class DecisionTreeRegression(Classifier):
    r"""Implementation of decision tree regression.

    Date:
        2024

    Author:
        Laurenz Farthofer

    License:
        MIT

    Documentation:
        https://scikit-learn.org/stable/modules/tree.html#regression

    See Also:
        * :class:`niaaml.classifiers.Classifier`
    """
    Name = "Decision Tree Regression"

    def __init__(self, **kwargs):
        r"""Initialize DecisionTree instance."""
        warnings.filterwarnings(action="ignore", category=ConvergenceWarning)
        warnings.filterwarnings(action="ignore", category=DataConversionWarning)
        warnings.filterwarnings(action="ignore", category=DataDimensionalityWarning)
        warnings.filterwarnings(action="ignore", category=EfficiencyWarning)
        warnings.filterwarnings(action="ignore", category=FitFailedWarning)
        warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)

        self._params = dict(
            criterion=ParameterDefinition(["squared_error", "friedman_mse", "absolute_error", "poisson"]),
            splitter=ParameterDefinition(["best", "random"]),
        )
        self.__decision_tree_regression = DTR()

    def set_parameters(self, **kwargs):
        r"""Set the parameters/arguments of the algorithm."""
        self.__decision_tree_regression.set_params(**kwargs)

    def fit(self, x, y, **kwargs):
        r"""Fit DecisionTree.

        Arguments:
            x (pandas.core.frame.DataFrame): n samples to classify.
            y (pandas.core.series.Series): n classes of the samples in the x array.

        Returns:
            None
        """
        self.__decision_tree_regression.fit(x, y)

    def predict(self, x, **kwargs):
        r"""Predict class for each sample (row) in x.

        Arguments:
            x (pandas.core.frame.DataFrame): n samples to classify.

        Returns:
            pandas.core.series.Series: n predicted classes.
        """
        return self.__decision_tree_regression.predict(x)

    def to_string(self):
        r"""User friendly representation of the object.

        Returns:
            str: User friendly representation of the object.
        """
        return Classifier.to_string(self).format(
            name=self.Name,
            args=self._parameters_to_string(
                self.__decision_tree_regression.get_params()
            ),
        )
