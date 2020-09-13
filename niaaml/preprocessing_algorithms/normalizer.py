from sklearn.preprocessing import Normalizer as nrm
from niaaml.preprocessing_algorithms.preprocessing_algorithm import PreprocessingAlgorithm

__all__ = ['Normalizer']

class Normalizer(PreprocessingAlgorithm):
    def process(self, x, **kwargs):
        return nrm.transform(x)
