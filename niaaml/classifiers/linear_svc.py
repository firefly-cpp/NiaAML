from niaaml.classifiers.classifier import Classifier
from niaaml.utilities import MinMax
from niaaml.utilities import ParameterDefinition
from sklearn.svm import LinearSVC as LSVC
import numpy as np

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

__all__ = ["LinearSVC"]


class LinearSVC(Classifier):
    r"""Implementation of linear support vector classification.

    Date:
        2020

    Author:
        Luka Pečnik

    License:
        MIT

    Reference:
        Fan, Rong-En, et al. "LIBLINEAR: A library for large linear classification." Journal of machine learning research 9.Aug (2008): 1871-1874.

    Documentation:
        https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html

    See Also:
        * :class:`niaaml.classifiers.Classifier`
    """
    Name = "Linear Support Vector Classification"

    def __init__(self, **kwargs):
        r"""Initialize LinearSVCClassifier instance."""
        warnings.filterwarnings(action="ignore", category=ChangedBehaviorWarning)
        warnings.filterwarnings(action="ignore", category=ConvergenceWarning)
        warnings.filterwarnings(action="ignore", category=DataConversionWarning)
        warnings.filterwarnings(action="ignore", category=DataDimensionalityWarning)
        warnings.filterwarnings(action="ignore", category=EfficiencyWarning)
        warnings.filterwarnings(action="ignore", category=FitFailedWarning)
        warnings.filterwarnings(action="ignore", category=NonBLASDotWarning)
        warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)

        self._params = dict(
            penalty=ParameterDefinition(["l1", "l2"]),
            max_iter=ParameterDefinition(MinMax(min=300, max=2000), np.uint),
        )
        self.__linear_SVC = LSVC()

    def set_parameters(self, **kwargs):
        r"""Set the parameters/arguments of the algorithm."""
        self.__linear_SVC.set_params(**kwargs)

    def fit(self, x, y, **kwargs):
        r"""Fit LinearSVCClassifier.

        Arguments:
            x (pandas.core.frame.DataFrame): n samples to classify.
            y (pandas.core.series.Series): n classes of the samples in the x array.

        Returns:
            None
        """
        self.__linear_SVC.fit(x, y)

    def predict(self, x, **kwargs):
        r"""Predict class for each sample (row) in x.

        Arguments:
           x (pandas.core.frame.DataFrame): n samples to classify.

        Returns:
            pandas.core.series.Series: n predicted classes.
        """
        return self.__linear_SVC.predict(x)

    def to_string(self):
        r"""User friendly representation of the object.

        Returns:
            str: User friendly representation of the object.
        """
        return Classifier.to_string(self).format(
            name=self.Name,
            args=self._parameters_to_string(self.__linear_SVC.get_params()),
        )
