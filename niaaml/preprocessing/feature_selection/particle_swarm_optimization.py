import sys
from NiaPy.algorithms.basic import ParticleSwarmOptimization as PSO
from NiaPy.task import StoppingTask
from sklearn.linear_model import LogisticRegression
from niaaml.preprocessing.feature_selection.feature_selection_algorithm import FeatureSelectionAlgorithm
from niaaml.utilities import ParameterDefinition, MinMax
import numpy
from niaaml.preprocessing.feature_selection.utility import _FeatureSelectionThresholdBenchmark

__all__ = [
    'ParticleSwarmOptimization'
]

class ParticleSwarmOptimization(FeatureSelectionAlgorithm):
    r"""Implementation of feature selection using PSO algorithm.

    Date:
        2020
    
    Author:
        Luka Pečnik 

    Reference:
        The implementation is adapted according to the following article:
        D. Fister, I. Fister, T. Jagrič, I. Fister Jr., J. Brest. A novel self-adaptive differential evolution for feature selection using threshold mechanism . In: Proceedings of the 2018 IEEE Symposium on Computational Intelligence (SSCI 2018), pp. 17-24, 2018.
    
    Reference URL: 
        http://iztok-jr-fister.eu/static/publications/236.pdf    

    License:
        MIT

    See Also:
        * :class:`niaaml.preprocessing.feature_selection.feature_selection_algorithm.FeatureSelectionAlgorithm`
    """
    Name = 'Particle Swarm Optimization'

    def __init__(self, **kwargs):
        r"""Initialize PSO feature selection algorithm.
        """
        self._params = dict(
            C1 = ParameterDefinition(MinMax(1.5, 2.5), param_type=float),
            C2 = ParameterDefinition(MinMax(1.5, 2.5), param_type=float)
        )
        self.__pso = PSO(NP=10)

    def set_parameters(self, **kwargs):
        r"""Set the parameters/arguments of the algorithm.
        """
        kwargs['NP'] = self.__pso.NP
        self.__pso.setParameters(**kwargs)

    def __final_output(self, sol):
        r"""Calculate final array of features.

        Arguments:
            sol (numpy.ndarray[float]): Individual of population/ possible solution.

        Returns:
            numpy.ndarray[bool]: Mask of selected features.
        """
        selected = numpy.ones(sol.shape[0] - 1, dtype=bool)
        threshold = sol[sol.shape[0] - 1]
        for i in range(sol.shape[0] - 1):
            if sol[i] < threshold:
                selected[i] = False
        return selected
    
    def select_features(self, x, y, **kwargs):
        r"""Perform the feature selection process.

        Arguments:
            x (numpy.ndarray[float]): Array of original features.
            y (Iterable[any]) Expected classifier results.

        Returns:
            numpy.ndarray[bool]: Mask of selected features.
        """
        num_features = x.shape[1]
        benchmark = _FeatureSelectionThresholdBenchmark(x, y)
        task = StoppingTask(D=num_features+1, nFES=1000, benchmark=benchmark)
        best = self.__pso.run(task)
        return self.__final_output(benchmark.get_best_solution())

    def to_string(self):
        r"""User friendly representation of the object.

        Returns:
            str: User friendly representation of the object.
        """
        return FeatureSelectionAlgorithm.to_string(self).format(name=self.Name, args=self._parameters_to_string(self.__pso.getParameters()))
