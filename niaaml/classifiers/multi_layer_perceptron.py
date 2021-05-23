from niaaml.classifiers.classifier import Classifier
from niaaml.utilities import ParameterDefinition
from sklearn.neural_network import MLPClassifier

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

__all__ = ["MultiLayerPerceptron"]


class MultiLayerPerceptron(Classifier):
    r"""Implementation of multi-layer perceptron classifier.

    Date:
        2020

    Author:
        Luka Pečnik

    License:
        MIT

    Reference:
        Glorot, Xavier, and Yoshua Bengio. “Understanding the difficulty of training deep feedforward neural networks.” International Conference on Artificial Intelligence and Statistics. 2010.

    Documentation:
        https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

    See Also:
        * :class:`niaaml.classifiers.Classifier`
    """
    Name = "Multi Layer Perceptron"

    def __init__(self, **kwargs):
        r"""Initialize MultiLayerPerceptron instance."""
        warnings.filterwarnings(action="ignore", category=ChangedBehaviorWarning)
        warnings.filterwarnings(action="ignore", category=ConvergenceWarning)
        warnings.filterwarnings(action="ignore", category=DataConversionWarning)
        warnings.filterwarnings(action="ignore", category=DataDimensionalityWarning)
        warnings.filterwarnings(action="ignore", category=EfficiencyWarning)
        warnings.filterwarnings(action="ignore", category=FitFailedWarning)
        warnings.filterwarnings(action="ignore", category=NonBLASDotWarning)
        warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)

        self._params = dict(
            activation=ParameterDefinition(["identity", "logistic", "tanh", "relu"]),
            solver=ParameterDefinition(["lbfgs", "sgd", "adam"]),
            learning_rate=ParameterDefinition(["constant", "invscaling", "adaptive"]),
        )
        self.__multi_layer_perceptron = MLPClassifier()

    def set_parameters(self, **kwargs):
        r"""Set the parameters/arguments of the algorithm."""
        self.__multi_layer_perceptron.set_params(**kwargs)

    def fit(self, x, y, **kwargs):
        r"""Fit MultiLayerPerceptron.

        Arguments:
            x (pandas.core.frame.DataFrame): n samples to classify.
            y (pandas.core.series.Series): n classes of the samples in the x array.

        Returns:
            None
        """
        self.__multi_layer_perceptron.fit(x, y)

    def predict(self, x, **kwargs):
        r"""Predict class for each sample (row) in x.

        Arguments:
            x (pandas.core.frame.DataFrame): n samples to classify.

        Returns:
            pandas.core.series.Series: n predicted classes.
        """
        return self.__multi_layer_perceptron.predict(x)

    def to_string(self):
        r"""User friendly representation of the object.

        Returns:
            str: User friendly representation of the object.
        """
        return Classifier.to_string(self).format(
            name=self.Name,
            args=self._parameters_to_string(self.__multi_layer_perceptron.get_params()),
        )
