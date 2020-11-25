from sklearn.preprocessing import Normalizer as Nrm
from niaaml.preprocessing.feature_transform import FeatureTransformAlgorithm
from niaaml.utilities import ParameterDefinition

__all__ = ['Normalizer']

class Normalizer(FeatureTransformAlgorithm):
    r"""Implementation of feature normalization algorithm.
    
    Date:
        2020

    Author
        Luka Pečnik

    License:
        MIT

    See Also:
        * :class:`niaaml.preprocessing.feature_transform.FeatureTransformAlgorithm`
    """

    def __init__(self, **kwargs):
        r"""Initialize SelectPercentile feature selection algorithm.
        """
        self._params = dict(
            norm = ParameterDefinition(['l1', 'l2', 'max'])
        )
        self.__params = None
        self.__normalizer = Nrm()

    def set_parameters(self, **kwargs):
        r"""Set the parameters/arguments of the algorithm.
        """
        self.__params = kwargs
        self.__params['axis'] = 0
    
	def fit(self, x, **kwargs):
		r"""Fit implemented transformation algorithm.

        Arguments:
            x (Iterable[any]): n samples to fit transformation algorithm.
		"""
		self.__normalizer.fit(x)

    def transform(self, x, **kwargs):
        r"""Transforms the given x data.

        Arguments:
            x (Iterable[any]): Data to transform.

        Returns:
            Iterable[any]: Transformed data.
        
        See Also:
            * :func:`niaaml.preprocessing.feature_transform.FeatureTransformAlgorithm.transform`
        """

        return self.__normalizer.transform(x)
