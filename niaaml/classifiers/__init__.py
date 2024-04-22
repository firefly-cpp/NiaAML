from niaaml.classifiers.classifier import Classifier
from niaaml.classifiers.random_forest import RandomForest
from niaaml.classifiers.multi_layer_perceptron import MultiLayerPerceptron
from niaaml.classifiers.linear_svc import LinearSVC
from niaaml.classifiers.ada_boost import AdaBoost
from niaaml.classifiers.extremely_randomized_trees import ExtremelyRandomizedTrees
from niaaml.classifiers.bagging import Bagging
from niaaml.classifiers.decision_tree import DecisionTree
from niaaml.classifiers.regression_decision_tree import DecisionTreeRegression
from niaaml.classifiers.k_neighbors import KNeighbors
from niaaml.classifiers.gaussian_process import GaussianProcess
from niaaml.classifiers.regression_gaussian_process import GaussianProcessRegression
from niaaml.classifiers.gaussian_naive_bayes import GaussianNB
from niaaml.classifiers.quadratic_driscriminant_analysis import (
    QuadraticDiscriminantAnalysis,
)
from niaaml.classifiers.regression_linear_model import LinearRegression
from niaaml.classifiers.regression_ridge import RidgeRegression
from niaaml.classifiers.regression_lasso import LassoRegression
from niaaml.classifiers.utility import ClassifierFactory

__all__ = [
    "Classifier",
    "RandomForest",
    "MultiLayerPerceptron",
    "LinearSVC",
    "AdaBoost",
    "Bagging",
    "ExtremelyRandomizedTrees",
    "DecisionTree",
    "DecisionTreeRegression",
    "KNeighbors",
    "GaussianProcess",
    "GaussianProcessRegression",
    "GaussianNB",
    "QuadraticDiscriminantAnalysis",
    "ClassifierFactory",
    "LinearRegression",
    "RidgeRegression",
    "LassoRegression",
]
