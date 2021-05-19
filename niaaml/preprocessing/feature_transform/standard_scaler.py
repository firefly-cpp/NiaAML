from sklearn.preprocessing import StandardScaler as StdScaler
from niaaml.preprocessing.feature_transform.feature_transform_algorithm import (
    FeatureTransformAlgorithm,
)

__all__ = ["StandardScaler"]


class StandardScaler(FeatureTransformAlgorithm):
    r"""Implementation of feature standard scaling algorithm.

    Date:
        2020

    Author:
        Luka Pečnik

    License:
        MIT

    Documentation:
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

    See Also:
        * :class:`niaaml.preprocessing.feature_transform.FeatureTransformAlgorithm`
    """
    Name = "Standard Scaler"

    def __init__(self, **kwargs):
        r"""Initialize StandardScaler."""
        super(StandardScaler, self).__init__()
        self.__std_scaler = StdScaler()

    def fit(self, x, **kwargs):
        r"""Fit implemented transformation algorithm.

        Arguments:
            x (pandas.core.frame.DataFrame): n samples to fit transformation algorithm.
        """
        self.__std_scaler.fit(x)

    def transform(self, x, **kwargs):
        r"""Transforms the given x data.

        Arguments:
            x (pandas.core.frame.DataFrame): Data to transform.

        Returns:
            pandas.core.frame.DataFrame: Transformed data.
        """

        return self.__std_scaler.transform(x)

    def to_string(self):
        r"""User friendly representation of the object.

        Returns:
            str: User friendly representation of the object.
        """
        return FeatureTransformAlgorithm.to_string(self).format(
            name=self.Name,
            args=self._parameters_to_string(self.__std_scaler.get_params()),
        )
