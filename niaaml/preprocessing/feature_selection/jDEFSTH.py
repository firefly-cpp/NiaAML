import sys
from NiaPy.algorithms.modified import SelfAdaptiveDifferentialEvolution
from NiaPy.task import StoppingTask
from NiaPy.benchmarks import Benchmark
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from niaaml.preprocessing.feature_selection.feature_selection_algorithm import FeatureSelectionAlgorithm

__all__ = [
    'jDEFSTH'
]

class jDEFSTH(FeatureSelectionAlgorithm):
    r"""Implementation of self-adaptive differential evolution for feature selection using threshold mechanism.

    Date:
        2020
    
    Author:
        Iztok Fister Jr.
    
    Reference:
        D. Fister, I. Fister, T. Jagrič, I. Fister Jr., J. Brest. A novel self-adaptive differential evolution for feature selection using threshold mechanism . In: Proceedings of the 2018 IEEE Symposium on Computational Intelligence (SSCI 2018), pp. 17-24, 2018.
    
    Reference URL: 
        http://iztok-jr-fister.eu/static/publications/236.pdf    

    License:
        MIT

	See Also:
		* :class:`niaaml.preprocessing.feature_selection.feature_selection_algorithm.FeatureSelectionAlgorithm`
    """

    def __final_output(self, sol):
        r"""Calculate final array of features.

        Arguments:
            sol (Iterable[float]): Individual of population/ possible solution.

        Returns:
            Iterable[bool]: Mask of selected features.
        """
        selected = numpy.ones(len(sol) - 1, dtype=bool)
        threshold = sol[len(sol)-1]
        for i in range(len(sol)-1):
            if sol[i] < threshold:
                selected[i] = False
        return selected
    
    def select_features(self, x, y, **kwargs):
        r"""Perform the feature selection process.

		Arguments:
			x (Iterable[any]): Array of original features.
            y (Iterable[any]) Expected classifier results.

		Returns:
			Iterable[bool]: Mask of selected features.
        """
        num_features = x.shape[1]
        algo = SelfAdaptiveDifferentialEvolution(NP=10, F=0.5, F_l=0.0, F_u=2.0, Tao1=0.9, CR=0.5, Tao2=0.45)
        benchmark = _FeatureSelectionThreshold(x, y)
        task = StoppingTask(D=num_features+1, nFES=1000, benchmark=benchmark)
        best = algo.run(task)
        return self.__final_output(benchmark.get_best_solution())

class _FeatureSelectionThreshold(Benchmark):
    r"""NiaPy Benchmark class implementation.

    Attributes:
        __best_fitness (float): Current best fitness of the optimization process.
        __best_solution (Iterable[float]): Current best solution of the optimization process.
    """
    
    def __init__(self, X, y):
        r"""Initialize feature selection benchmark.

		Arguments:
            X (Iterable[any]): Features.
            y (Iterable[any]) Expected classifier results.
        """
        self.__best_fitness = float('inf')
        self.__best_solution = None
        Benchmark.__init__(self, 0.0, 1.0)
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(X, y, test_size=0.2)

    def get_best_solution(self):
        r"""Get best solution found.

        Returns:
            Iterable[float]: Best solution found.
        """
        return self.__best_solution

    def function(self):
        r"""Override Benchmark function.

        Returns:
            Callable[[int, Iterable[float]], float]: Fitness evaluation function.
        """

        def evaluate(D, sol):
            r"""Evaluate features.

            Arguments:
                D (uint): Number of dimensions.
                sol (Iterable[float]): Individual of population/ possible solution.
            
            Returns:
                float: Fitness.
            """
            selected = []  #array for selected features
            self.Threshold = sol[D-1]  # current threshold
            
            # select features
            for i in range(len(sol)-1):
                if sol[i] < self.Threshold:
                    pass
                else:
                    selected.append(i)

            # in the case if threshold is too low (no features selected)
            if len(selected) == 0:
                return 1
            
            lr = LogisticRegression(solver='lbfgs', max_iter=10000).fit(
                self.train_X[:, selected], self.train_y)
            accuracy = lr.score(self.test_X[:, selected], self.test_y)
            fitness = 1.0 - accuracy
            
            if fitness < self.__best_fitness:
                self.__best_fitness = fitness
                self.__best_solution = sol
            return fitness

        return evaluate
