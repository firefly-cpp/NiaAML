__all__ = ["FeatureEncoder"]


class FeatureEncoder:
    r"""Class for implementing feature encoders.

    Date:
        2020

    Author:
        Luka Pečnik

    License:
        MIT

    Attributes:
        Name (str): Name of the feature encoder.
    """
    Name = None

    def __init__(self, **kwargs):
        r"""Initialize feature encoder."""
        return None

    def fit(self, feature):
        r"""Fit feature encoder.

        Arguments:
            feature (pandas.core.frame.DataFrame): A column (categorical) from DataFrame of features.
        """
        return None

    def transform(self, feature):
        r"""Transform feature's values.

        Arguments:
            feature (pandas.core.frame.DataFrame): A column (categorical) from DataFrame of features.

        Returns:
            pandas.core.frame.DataFrame: A transformed column.
        """
        return None

    def to_string(self):
        r"""User friendly representation of the object.

        Returns:
            str: User friendly representation of the object.
        """
        return "{name}"
