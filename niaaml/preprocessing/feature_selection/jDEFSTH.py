import sys
from NiaPy.algorithms.modified import SelfAdaptiveDifferentialEvolution
from NiaPy.task import StoppingTask
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from niaaml.preprocessing.feature_selection.feature_selection_algorithm import FeatureSelectionAlgorithm
from niaaml.preprocessing.feature_selection.utility import _FeatureSelectionThresholdBenchmark
import numpy

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
    Name = 'Self-Adaptive Differential Evolution'

    def __init__(self, **kwargs):
        r"""Initialize GWO feature selection algorithm.
        """
        super(jDEFSTH, self).__init__()
        self.__jdefsth = SelfAdaptiveDifferentialEvolution(NP=10, F=0.5, F_l=0.0, F_u=2.0, Tao1=0.9, CR=0.5, Tao2=0.45)

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
        best = self.__jdefsth.run(task)
        return self.__final_output(benchmark.get_best_solution())

    def to_string(self):
        r"""User friendly representation of the object.

        Returns:
            str: User friendly representation of the object.
        """
        return FeatureSelectionAlgorithm.to_string(self).format(name=self.Name, args=self._parameters_to_string(self.__jdefsth.getParameters()))
