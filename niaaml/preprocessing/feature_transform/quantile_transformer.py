from sklearn.preprocessing import QuantileTransformer as QT
from niaaml.preprocessing.feature_transform.feature_transform_algorithm import (
    FeatureTransformAlgorithm,
)
from niaaml.utilities import ParameterDefinition

__all__ = ["QuantileTransformer"]


class QuantileTransformer(FeatureTransformAlgorithm):
    r"""Implementation of quantile transformer.

    Date:
        2020

    Author:
        Luka Pečnik

    License:
        MIT

    Documentation:
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html#sklearn.preprocessing.QuantileTransformer

    See Also:
        * :class:`niaaml.preprocessing.feature_transform.FeatureTransformAlgorithm`
    """
    Name = "Quantile Transformer"

    def __init__(self, **kwargs):
        r"""Initialize QuantileTransformer."""
        self._params = dict(
            output_distribution=ParameterDefinition(["uniform", "normal"])
        )
        self.__quantile_transformer = QT()

    def fit(self, x, **kwargs):
        r"""Fit implemented transformation algorithm.

        Arguments:
            x (pandas.core.frame.DataFrame): n samples to fit transformation algorithm.
        """
        self.__quantile_transformer.fit(x)

    def transform(self, x, **kwargs):
        r"""Transforms the given x data.

        Arguments:
            x (pandas.core.frame.DataFrame): Data to transform.

        Returns:
            pandas.core.frame.DataFrame: Transformed data.
        """

        return self.__quantile_transformer.transform(x)

    def to_string(self):
        r"""User friendly representation of the object.

        Returns:
            str: User friendly representation of the object.
        """
        return FeatureTransformAlgorithm.to_string(self).format(
            name=self.Name,
            args=self._parameters_to_string(self.__quantile_transformer.get_params()),
        )
