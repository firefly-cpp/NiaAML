import numpy as np
from niapy.problems import Problem
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

__all__ = ["_FeatureSelectionThresholdProblem"]


class _FeatureSelectionThresholdProblem(Problem):
    r"""NiaPy Problem class implementation.

    Attributes:
        __best_fitness (float): Current best fitness of the optimization process.
        __best_solution (numpy.ndarray[float]): Current best solution of the optimization process.
    """

    def __init__(self, X, y):
        r"""Initialize feature selection problem.

        Arguments:
            X (pandas.core.frame.DataFrame): Features.
            y (pandas.core.series.Series) Expected classifier results.
        """
        self.__best_fitness = np.inf
        self.__best_solution = None
        super().__init__(X.shape[1] + 1, 0.0, 1.0)
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(
            X, y, test_size=0.2
        )

    def get_best_solution(self):
        r"""Get best solution found.

        Returns:
            numpy.ndarray[float]: Best solution found.
        """
        return self.__best_solution

    def _evaluate(self, x):
        r"""Override fitness function.

        Args:
            x (np.ndarray): Solution vector.

        Returns:
            float: Fitness value of `x`.
        """
        self.Threshold = x[-1]  # current threshold

        # select features
        selected = x[:-1] >= self.Threshold

        # in the case if threshold is too low (no features selected)
        if np.sum(selected) == 0:
            return 1

        lr = LogisticRegression(solver="lbfgs", max_iter=10000).fit(
            self.train_X.iloc[:, selected], self.train_y
        )
        accuracy = lr.score(self.test_X.iloc[:, selected], self.test_y)
        fitness = 1.0 - accuracy

        if fitness < self.__best_fitness:
            self.__best_fitness = fitness
            self.__best_solution = x
        return fitness
