from niaaml.classifiers.classifier import Classifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

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

__all__ = ["QuadraticDiscriminantAnalysis"]


class QuadraticDiscriminantAnalysis(Classifier):
    r"""Implementation of quadratic discriminant analysis classifier.

    Date:
        2020

    Author:
        Luka Pečnik

    License:
        MIT

    Reference:
        “The Elements of Statistical Learning”, Hastie T., Tibshirani R., Friedman J., Section 4.3, p.106-119, 2008.

    Documentation:
        https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html#sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis

    See Also:
        * :class:`niaaml.classifiers.Classifier`
    """
    Name = "Quadratic Discriminant Analysis"

    def __init__(self, **kwargs):
        r"""Initialize QuadraticDiscriminantAnalysis instance."""
        warnings.filterwarnings(action="ignore", category=ChangedBehaviorWarning)
        warnings.filterwarnings(action="ignore", category=ConvergenceWarning)
        warnings.filterwarnings(action="ignore", category=DataConversionWarning)
        warnings.filterwarnings(action="ignore", category=DataDimensionalityWarning)
        warnings.filterwarnings(action="ignore", category=EfficiencyWarning)
        warnings.filterwarnings(action="ignore", category=FitFailedWarning)
        warnings.filterwarnings(action="ignore", category=NonBLASDotWarning)
        warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)

        self.__qda = QDA()
        super(QuadraticDiscriminantAnalysis, self).__init__()

    def set_parameters(self, **kwargs):
        r"""Set the parameters/arguments of the algorithm."""
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
        return Classifier.to_string(self).format(
            name=self.Name, args=self._parameters_to_string(self.__qda.get_params())
        )
