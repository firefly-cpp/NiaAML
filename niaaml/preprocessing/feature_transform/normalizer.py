from sklearn.preprocessing import normalize
from niaaml.preprocessing.feature_transform import FeatureTransformAlgorithm

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

    def transform(self, x, **kwargs):
        r"""Transforms the given x data.

        Arguments:
            x (Iterable[any]): Data to transform.

        Returns:
            Iterable[any]: Transformed data.
        
        See Also:
            * :func:`niaaml.preprocessing.feature_transform.FeatureTransformAlgorithm.transform`
        """

        return normalize(x, axis=0)
