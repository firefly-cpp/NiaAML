import sys
import random
from NiaPy.algorithms.modified import SelfAdaptiveDifferentialEvolution
from NiaPy.task import StoppingTask
from NiaPy.benchmarks import Benchmark
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_wine

__all__ = [
    'jDEFSTH'
]

#globals
best_fitness = sys.maxsize
best_solution = None

class jDEFSTH(object):
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
    """

    def __init__(self, **kwargs):
        r"""Initialize the jDEFSTH.
        """
        self._set_parameters(**kwargs) #TODO

    def _set_parameters(self, **kwargs):
        r"""Set the parameters/arguments of the algorithm.
        """
        return

    def final_output(self, sol):
        selected = []
        threshold = sol[len(sol)-1]
        for i in range(len(sol)-1):
            if sol[i] < threshold:
                pass
            else:
                selected.append(i)
        return selected
    
    def select_features(self, x, y, **kwargs):
        num_features = X.shape[1]
        algo = SelfAdaptiveDifferentialEvolution(NP=10, F=0.5, F_l=0.0, F_u=2.0, Tao1=0.9, CR=0.5, Tao2=0.45)
        task = StoppingTask(D=num_features+1, nFES=1000, benchmark=FeatureSelectionThreshold(X, y))
        best = algo.run(task)
        return self.final_output(best[0])


class FeatureSelectionThreshold(Benchmark):
    def __init__(self, X, y):
        Benchmark.__init__(self, 0.0, 1.0)
        self.Threshold = random.uniform(0, 1)
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(
            X, y, test_size=0.2)

    def function(self):  # eval. func. for NiaPy
        def evaluate(D, sol):
            selected = []  #array for selected features
            self.Threshold = sol[D-1]  # current threshold
            
            global best_fitness
            global best_solution
            
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
            
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = selected
            return fitness

        return evaluate

#test
#X, y = load_wine(return_X_y=True)
#a = jDEFSTH()
#a.select_features(X,y)
