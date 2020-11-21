from sklearn.preprocessing import normalize
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

    def _set_parameters(self, **kwargs):
        r"""Set the parameters/arguments of the algorithm.
        """
        self.__params = kwargs
        self.__params['axis'] = 0

    def transform(self, x, **kwargs):
        r"""Transforms the given x data.

        Arguments:
            x (Iterable[any]): Data to transform.

        Returns:
            Iterable[any]: Transformed data.
        
        See Also:
            * :func:`niaaml.preprocessing.feature_transform.FeatureTransformAlgorithm.transform`
        """

        return normalize(x, **self.__params)
