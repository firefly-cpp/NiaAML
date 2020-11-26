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
    Name = 'Standard Scaler'

    def __init__(self, **kwargs):
        r"""Initialize SelectPercentile feature selection algorithm.
        """
        self.__std_scaler = StdScaler()

    def fit(self, x, **kwargs):
        r"""Fit implemented transformation algorithm.

        Arguments:
            x (numpy.ndarray[float]): n samples to fit transformation algorithm.
        """
        self.__std_scaler.fit(x)

    def transform(self, x, **kwargs):
        r"""Transforms the given x data.

        Arguments:
            x (numpy.ndarray[float]): Data to transform.

        Returns:
            numpy.ndarray[float]: Transformed data.
        """
        
        return self.__std_scaler.transform(x)

    def to_string(self):
        r"""User friendly representation of the object.

        Returns:
            str: User friendly representation of the object.
        """
        return FeatureTransformAlgorithm.to_string(self).format(name=self.Name, args=self._parameters_to_string(self.__std_scaler.get_params()))