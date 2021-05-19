from sklearn.impute import SimpleImputer as SI
from niaaml.preprocessing.imputation.imputer import Imputer
import numpy as np
import pandas as pd

__all__ = ["SimpleImputer"]


class SimpleImputer(Imputer):
    r"""Implementation of simple imputer.

    Date:
        2020

    Author:
        Luka Pečnik

    License:
        MIT

    Documentation:
        https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html

    See Also:
        * :class:`niaaml.preprocessing.imputation.Imputer`
    """
    Name = "Simple Imputer"

    def __init__(self, **kwargs):
        r"""Initialize imputer."""
        self.__simple_imputer = SI(missing_values=np.nan)

    def fit(self, feature):
        r"""Fit imputer.

        Arguments:
            feature (pandas.core.frame.DataFrame): A column from DataFrame of features.
        """
        if not pd.api.types.is_numeric_dtype(feature.iloc[:, 0]):
            replacement_val = feature.mode().iloc[0, 0]
            self.__simple_imputer.set_params(
                **{"fill_value": replacement_val, "strategy": "constant"}
            )
            self.__simple_imputer.fit(feature)
        else:
            self.__simple_imputer.fit(feature)

    def transform(self, feature):
        r"""Transform feature's values.

        Arguments:
            feature (pandas.core.frame.DataFrame): A column from DataFrame of features.

        Returns:
            pandas.core.frame.DataFrame: A transformed column.
        """
        return self.__simple_imputer.transform(feature)

    def to_string(self):
        r"""User friendly representation of the object.

        Returns:
            str: User friendly representation of the object.
        """
        return Imputer.to_string(self).format(name=self.Name)
