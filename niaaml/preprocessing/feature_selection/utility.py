from niaaml.utilities import Factory
import niaaml.preprocessing.feature_selection as fs
from NiaPy.benchmarks import Benchmark

__all__ = [
    'FeatureSelectionAlgorithmFactory',
    '_FeatureSelectionThresholdBenchmark'
]

class FeatureSelectionAlgorithmFactory(Factory):
    r"""Class with string mappings to feature selection algorithms.

    Attributes:
        _entities (Dict[str, FeatureSelectionAlgorithm]): Mapping from strings to feature selection algorithms.
    
    See Also:
        * :class:`niaaml.utilities.Factory`
    """

    def _set_parameters(self, **kwargs):
        r"""Set the parameters/arguments of the factory.
        """
        self._entities = {
            'jDEFSTH': fs.jDEFSTH,
            'SelectKBest': fs.SelectKBest,
            'SelectPercentile': fs.SelectPercentile,
            'VarianceThreshold': fs.VarianceThreshold,
            'BatAlgorithm': fs.BatAlgorithm,
            'DifferentialEvolution': fs.DifferentialEvolution,
            'GreyWolfOptimizer': fs.GreyWolfOptimizer,
            'ParticleSwarmOptimization': fs.ParticleSwarmOptimization
        }

class _FeatureSelectionThresholdBenchmark(Benchmark):
    r"""NiaPy Benchmark class implementation.

    Attributes:
        __best_fitness (float): Current best fitness of the optimization process.
        __best_solution (numpy.ndarray[float]): Current best solution of the optimization process.
    """
    
    def __init__(self, X, y):
        r"""Initialize feature selection benchmark.

        Arguments:
            X (numpy.ndarray[float]): Features.
            y (Iterable[any]) Expected classifier results.
        """
        self.__best_fitness = float('inf')
        self.__best_solution = None
        Benchmark.__init__(self, 0.0, 1.0)
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(X, y, test_size=0.2)

    def get_best_solution(self):
        r"""Get best solution found.

        Returns:
            numpy.ndarray[float]: Best solution found.
        """
        return self.__best_solution

    def function(self):
        r"""Override Benchmark function.

        Returns:
            Callable[[int, numpy.ndarray[float]], float]: Fitness evaluation function.
        """

        def evaluate(D, sol):
            r"""Evaluate features.

            Arguments:
                D (uint): Number of dimensions.
                sol (numpy.ndarray[float]): Individual of population/ possible solution.
            
            Returns:
                float: Fitness.
            """
            selected = []  #array for selected features
            self.Threshold = sol[D-1]  # current threshold
            
            # select features
            for i in range(sol.shape[0] - 1):
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