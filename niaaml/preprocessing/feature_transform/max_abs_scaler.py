from sklearn.preprocessing import MaxAbsScaler as MAS
from niaaml.preprocessing.feature_transform.feature_transform_algorithm import (
    FeatureTransformAlgorithm,
)

__all__ = ["MaxAbsScaler"]


class MaxAbsScaler(FeatureTransformAlgorithm):
    r"""Implementation of feature scaling by its maximum absolute value.

    Date:
        2020

    Author:
        Luka Pečnik

    License:
        MIT

    Documentation:
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html#sklearn.preprocessing.MaxAbsScaler

    See Also:
        * :class:`niaaml.preprocessing.feature_transform.FeatureTransformAlgorithm`
    """
    Name = "Maximum Absolute Scaler"

    def __init__(self, **kwargs):
        r"""Initialize MaxAbsScaler."""
        super(MaxAbsScaler, self).__init__()
        self.__max_abs_scaler = MAS()

    def fit(self, x, **kwargs):
        r"""Fit implemented transformation algorithm.

        Arguments:
            x (pandas.core.frame.DataFrame): n samples to fit transformation algorithm.
        """
        self.__max_abs_scaler.fit(x)

    def transform(self, x, **kwargs):
        r"""Transforms the given x data.

        Arguments:
            x (pandas.core.frame.DataFrame): Data to transform.

        Returns:
            pandas.core.frame.DataFrame: Transformed data.
        """

        return self.__max_abs_scaler.transform(x)

    def to_string(self):
        r"""User friendly representation of the object.

        Returns:
            str: User friendly representation of the object.
        """
        return FeatureTransformAlgorithm.to_string(self).format(
            name=self.Name,
            args=self._parameters_to_string(self.__max_abs_scaler.get_params()),
        )
