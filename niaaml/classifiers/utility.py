from niaaml.classifiers.regression_decision_tree import DecisionTreeRegression
from niaaml.classifiers.regression_gaussian_process import GaussianProcessRegression
from niaaml.utilities import Factory
from niaaml.classifiers.ada_boost import AdaBoost
from niaaml.classifiers.bagging import Bagging
from niaaml.classifiers.extremely_randomized_trees import ExtremelyRandomizedTrees
from niaaml.classifiers.linear_svc import LinearSVC
from niaaml.classifiers.multi_layer_perceptron import MultiLayerPerceptron
from niaaml.classifiers.random_forest import RandomForest
from niaaml.classifiers.decision_tree import DecisionTree
from niaaml.classifiers.k_neighbors import KNeighbors
from niaaml.classifiers.gaussian_process import GaussianProcess
from niaaml.classifiers.gaussian_naive_bayes import GaussianNB
from niaaml.classifiers.quadratic_driscriminant_analysis import (
    QuadraticDiscriminantAnalysis,
)
from niaaml.classifiers.regression_linear_model import LinearRegression
from niaaml.classifiers.regression_ridge import RidgeRegression
from niaaml.classifiers.regression_lasso import LassoRegression

__all__ = ["ClassifierFactory"]


class ClassifierFactory(Factory):
    r"""Class with string mappings to classifiers.

    Date:
        2020

    Author:
        Luka Pečnik

    License:
        MIT

    Attributes:
        _entities (Dict[str, Classifier]): Mapping from strings to classifiers.

    See Also:
        * :class:`niaaml.utilities.Factory`
    """

    def _set_parameters(self, **kwargs):
        r"""Set the parameters/arguments of the factory."""
        self._entities = {
            "AdaBoost": AdaBoost,
            "Bagging": Bagging,
            "ExtremelyRandomizedTrees": ExtremelyRandomizedTrees,
            "LinearSVC": LinearSVC,
            "MultiLayerPerceptron": MultiLayerPerceptron,
            "RandomForest": RandomForest,
            "DecisionTree": DecisionTree,
            "DecisionTreeRegression": DecisionTreeRegression,
            "KNeighbors": KNeighbors,
            "GaussianProcess": GaussianProcess,
            "GaussianProcessRegression": GaussianProcessRegression,
            "GaussianNB": GaussianNB,
            "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis,
            "LinearRegression": LinearRegression,
            "RidgeRegression": RidgeRegression,
            "LassoRegression": LassoRegression,
        }
