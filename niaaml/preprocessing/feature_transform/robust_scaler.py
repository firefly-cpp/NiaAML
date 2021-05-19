from sklearn.preprocessing import RobustScaler as RS
from niaaml.preprocessing.feature_transform.feature_transform_algorithm import (
    FeatureTransformAlgorithm,
)
from niaaml.utilities import ParameterDefinition

__all__ = ["RobustScaler"]


class RobustScaler(FeatureTransformAlgorithm):
    r"""Implementation of the robust scaler.

    Date:
        2020

    Author:
        Luka Pečnik

    License:
        MIT

    Documentation:
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler

    See Also:
        * :class:`niaaml.preprocessing.feature_transform.FeatureTransformAlgorithm`
    """
    Name = "Robust Scaler"

    def __init__(self, **kwargs):
        r"""Initialize RobustScaler."""
        self._params = dict(
            with_centering=ParameterDefinition([True, False]),
            with_scaling=ParameterDefinition([True, False]),
        )
        self.__robust_scaler = RS()

    def fit(self, x, **kwargs):
        r"""Fit implemented transformation algorithm.

        Arguments:
            x (pandas.core.frame.DataFrame): n samples to fit transformation algorithm.
        """
        self.__robust_scaler.fit(x)

    def transform(self, x, **kwargs):
        r"""Transforms the given x data.

        Arguments:
            x (pandas.core.frame.DataFrame): Data to transform.

        Returns:
            pandas.core.frame.DataFrame: Transformed data.
        """

        return self.__robust_scaler.transform(x)

    def to_string(self):
        r"""User friendly representation of the object.

        Returns:
            str: User friendly representation of the object.
        """
        return FeatureTransformAlgorithm.to_string(self).format(
            name=self.Name,
            args=self._parameters_to_string(self.__robust_scaler.get_params()),
        )
