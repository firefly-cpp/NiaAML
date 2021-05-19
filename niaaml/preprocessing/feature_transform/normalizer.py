from sklearn.preprocessing import Normalizer as Nrm
from niaaml.preprocessing.feature_transform.feature_transform_algorithm import (
    FeatureTransformAlgorithm,
)
from niaaml.utilities import ParameterDefinition

__all__ = ["Normalizer"]


class Normalizer(FeatureTransformAlgorithm):
    r"""Implementation of feature normalization algorithm.

    Date:
        2020

    Author:
        Luka Pečnik

    License:
        MIT

    Documentation:
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer

    See Also:
        * :class:`niaaml.preprocessing.feature_transform.FeatureTransformAlgorithm`
    """
    Name = "Normalizer"

    def __init__(self, **kwargs):
        r"""Initialize Normalizer."""
        self._params = dict(norm=ParameterDefinition(["l1", "l2", "max"]))
        self.__params = None
        self.__normalizer = Nrm()

    def set_parameters(self, **kwargs):
        r"""Set the parameters/arguments of the algorithm."""
        self.__params = kwargs
        self.__params["axis"] = 0

    def fit(self, x, **kwargs):
        r"""Fit implemented transformation algorithm.

        Arguments:
            x (pandas.core.frame.DataFrame): n samples to fit transformation algorithm.
        """
        self.__normalizer.fit(x)

    def transform(self, x, **kwargs):
        r"""Transforms the given x data.

        Arguments:
            x (pandas.core.frame.DataFrame): Data to transform.

        Returns:
            pandas.core.frame.DataFrame: Transformed data.
        """
        return self.__normalizer.transform(x)

    def to_string(self):
        r"""User friendly representation of the object.

        Returns:
            str: User friendly representation of the object.
        """
        return FeatureTransformAlgorithm.to_string(self).format(
            name=self.Name,
            args=self._parameters_to_string(self.__normalizer.get_params()),
        )
