from niaaml.classifiers.classifier import Classifier
from sklearn.linear_model import LinearRegression as LR

import warnings
from sklearn.exceptions import (
    ConvergenceWarning,
    DataConversionWarning,
    DataDimensionalityWarning,
    EfficiencyWarning,
    FitFailedWarning,
    UndefinedMetricWarning,
)

from niaaml.utilities import ParameterDefinition

__all__ = ["LinearRegression"]


class LinearRegression(Classifier):
    r"""Implementation of linear regression.

    Date:
        2024

    Author:
        Laurenz Farthofer

    License:
        MIT

    Documentation:
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression

    See Also:
        * :class:`niaaml.classifiers.Classifier`
    """
    Name = "Linear Regression"
    Task = "Regression"

    def __init__(self, **kwargs):
        r"""Initialize LinearRegression instance."""
        warnings.filterwarnings(action="ignore", category=ConvergenceWarning)
        warnings.filterwarnings(action="ignore", category=DataConversionWarning)
        warnings.filterwarnings(action="ignore", category=DataDimensionalityWarning)
        warnings.filterwarnings(action="ignore", category=EfficiencyWarning)
        warnings.filterwarnings(action="ignore", category=FitFailedWarning)
        warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)

        self.model = LR()

        self._params = dict(
            fit_intercept=ParameterDefinition([True, False]),
        )

    def set_parameters(self, **kwargs):
        r"""Set the parameters/arguments of the algorithm."""
        self.model.set_params(**kwargs)

    def fit(self, x, y, **kwargs):
        r"""Fit LinearSVCClassifier.

        Arguments:
            x (pandas.core.frame.DataFrame): n samples to classify.
            y (pandas.core.series.Series): n classes of the samples in the x array.

        Returns:
            None
        """
        self.model.fit(x, y)

    def predict(self, x, **kwargs):
        r"""Predict class for each sample (row) in x.

        Arguments:
            x (pandas.core.frame.DataFrame): n samples to classify.

        Returns:
            pandas.core.series.Series: n predicted classes.
        """
        return self.model.predict(x)

    def to_string(self):
        r"""User friendly representation of the object.

        Returns:
            str: User friendly representation of the object.
        """
        return Classifier.to_string(self).format(
            name=self.Name,
            args=self._parameters_to_string(self.__gaussian_process.get_params()),
        )
