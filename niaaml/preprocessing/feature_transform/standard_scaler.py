from sklearn.preprocessing import StandardScaler as StdScaler
from niaaml.preprocessing.feature_transform import FeatureTransformAlgorithm

__all__ = ['StandardScaler']

class StandardScaler(FeatureTransformAlgorithm):
    r"""Implementation of feature standard scaling algorithm.
    
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
        
        return StdScaler().fit_transform(x)
