from niaaml.preprocessing.encoding.feature_encoder import FeatureEncoder
from sklearn.preprocessing import OneHotEncoder as OHE
import pandas as pd

__all__ = ["OneHotEncoder"]


class OneHotEncoder(FeatureEncoder):
    r"""Implementation of one-hot encoder.

    Date:
        2020

    Author:
        Luka Pečnik

    License:
        MIT

    Reference:
        Seger, Cedric. "An investigation of categorical variable encoding techniques in machine learning: binary versus one-hot and feature hashing." (2018).

    Documentation:
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

    See Also:
        * :class:`niaaml.preprocessing.encoding.FeatureEncoder`
    """
    Name = "One-Hot Encoder"

    def __init__(self, **kwargs):
        r"""Initialize feature encoder."""
        self.__one_hot_encoder = OHE(handle_unknown="ignore")

    def fit(self, feature):
        r"""Fit feature encoder.

        Arguments:
            feature (pandas.core.frame.DataFrame): A column (categorical) from DataFrame of features.
        """
        self.__one_hot_encoder.fit(feature)

    def transform(self, feature):
        r"""Transform feature's values.

        Arguments:
            feature (pandas.core.frame.DataFrame): A column (categorical) from DataFrame of features.

        Returns:
            pandas.core.frame.DataFrame: A transformed column.
        """
        return pd.DataFrame(self.__one_hot_encoder.transform(feature).toarray())

    def to_string(self):
        r"""User friendly representation of the object.

        Returns:
            str: User friendly representation of the object.
        """
        return FeatureEncoder.to_string(self).format(name=self.Name)
