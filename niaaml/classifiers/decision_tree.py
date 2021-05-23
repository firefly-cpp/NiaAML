from niaaml.classifiers.classifier import Classifier
from niaaml.utilities import ParameterDefinition
from sklearn.tree import DecisionTreeClassifier as DTC

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

__all__ = ["DecisionTree"]


class DecisionTree(Classifier):
    r"""Implementation of decision tree classifier.

    Date:
        2020

    Author:
        Luka Pečnik

    License:
        MIT

    Reference:
        L. Breiman, J. Friedman, R. Olshen, and C. Stone, “Classification and Regression Trees”, Wadsworth, Belmont, CA, 1984.

    Documentation:
        https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier

    See Also:
        * :class:`niaaml.classifiers.Classifier`
    """
    Name = "Decision Tree Classifier"

    def __init__(self, **kwargs):
        r"""Initialize DecisionTree instance."""
        warnings.filterwarnings(action="ignore", category=ChangedBehaviorWarning)
        warnings.filterwarnings(action="ignore", category=ConvergenceWarning)
        warnings.filterwarnings(action="ignore", category=DataConversionWarning)
        warnings.filterwarnings(action="ignore", category=DataDimensionalityWarning)
        warnings.filterwarnings(action="ignore", category=EfficiencyWarning)
        warnings.filterwarnings(action="ignore", category=FitFailedWarning)
        warnings.filterwarnings(action="ignore", category=NonBLASDotWarning)
        warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)

        self._params = dict(
            criterion=ParameterDefinition(["gini", "entropy"]),
            splitter=ParameterDefinition(["best", "random"]),
        )
        self.__decision_tree_classifier = DTC()

    def set_parameters(self, **kwargs):
        r"""Set the parameters/arguments of the algorithm."""
        self.__decision_tree_classifier.set_params(**kwargs)

    def fit(self, x, y, **kwargs):
        r"""Fit DecisionTree.

        Arguments:
            x (pandas.core.frame.DataFrame): n samples to classify.
            y (pandas.core.series.Series): n classes of the samples in the x array.

        Returns:
            None
        """
        self.__decision_tree_classifier.fit(x, y)

    def predict(self, x, **kwargs):
        r"""Predict class for each sample (row) in x.

        Arguments:
            x (pandas.core.frame.DataFrame): n samples to classify.

        Returns:
            pandas.core.series.Series: n predicted classes.
        """
        return self.__decision_tree_classifier.predict(x)

    def to_string(self):
        r"""User friendly representation of the object.

        Returns:
            str: User friendly representation of the object.
        """
        return Classifier.to_string(self).format(
            name=self.Name,
            args=self._parameters_to_string(
                self.__decision_tree_classifier.get_params()
            ),
        )
