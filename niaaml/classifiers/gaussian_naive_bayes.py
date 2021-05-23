from niaaml.classifiers.classifier import Classifier
from sklearn.naive_bayes import GaussianNB as GNB

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

__all__ = ["GaussianNB"]


class GaussianNB(Classifier):
    r"""Implementation of gaussian Naive Bayes classifier.

    Date:
        2020

    Author:
        Luka Pečnik

    License:
        MIT

    Reference:
        Murphy, Kevin P. "Naive bayes classifiers." University of British Columbia 18 (2006): 60.

    Documentation:
        https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html

    See Also:
        * :class:`niaaml.classifiers.Classifier`
    """
    Name = "Gaussian Naive Bayes"

    def __init__(self, **kwargs):
        r"""Initialize GaussianNB instance."""
        warnings.filterwarnings(action="ignore", category=ChangedBehaviorWarning)
        warnings.filterwarnings(action="ignore", category=ConvergenceWarning)
        warnings.filterwarnings(action="ignore", category=DataConversionWarning)
        warnings.filterwarnings(action="ignore", category=DataDimensionalityWarning)
        warnings.filterwarnings(action="ignore", category=EfficiencyWarning)
        warnings.filterwarnings(action="ignore", category=FitFailedWarning)
        warnings.filterwarnings(action="ignore", category=NonBLASDotWarning)
        warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)

        self.__gaussian_nb = GNB()
        super(GaussianNB, self).__init__()

    def set_parameters(self, **kwargs):
        r"""Set the parameters/arguments of the algorithm."""
        self.__gaussian_nb.set_params(**kwargs)

    def fit(self, x, y, **kwargs):
        r"""Fit GaussianNB.

        Arguments:
            x (pandas.core.frame.DataFrame): n samples to classify.
            y (pandas.core.series.Series): n classes of the samples in the x array.

        Returns:
            None
        """
        self.__gaussian_nb.fit(x, y)

    def predict(self, x, **kwargs):
        r"""Predict class for each sample (row) in x.

        Arguments:
            x (pandas.core.frame.DataFrame): n samples to classify.

        Returns:
            pandas.core.series.Series: n predicted classes.
        """
        return self.__gaussian_nb.predict(x)

    def to_string(self):
        r"""User friendly representation of the object.

        Returns:
            str: User friendly representation of the object.
        """
        return Classifier.to_string(self).format(
            name=self.Name,
            args=self._parameters_to_string(self.__gaussian_nb.get_params()),
        )
