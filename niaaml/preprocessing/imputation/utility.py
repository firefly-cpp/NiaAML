from niaaml.preprocessing.imputation.simple_imputer import SimpleImputer
from niaaml.utilities import Factory

__all__ = ["ImputerFactory", "impute_features"]


def impute_features(features, imputer):
    """Impute features with missing data.

    Arguments:
        features (pandas.core.frame.DataFrame): DataFrame of features.
        imputer (str): Name of the imputer to use.

    Returns:
                Tuple[pandas.core.frame.DataFrame, Dict[Imputer]]:
                        1. Converted dataframe.
                        2. Dictionary of imputers for all features with missing data.
    """
    imp = ImputerFactory().get_result(imputer)

    imputers = {}
    cols = [col for col in features.columns if features[col].isnull().any()]
    for c in cols:
        imp.fit(features[[c]])
        features.loc[:, c] = imp.transform(features[[c]])
        imputers[c] = imp

    return features, imputers if len(imputers) > 0 else None


class ImputerFactory(Factory):
    r"""Class with string mappings to imputers.

    Attributes:
        _entities (Dict[str, Imputer]): Mapping from strings to imputers.

    See Also:
        * :class:`niaaml.utilities.Factory`
    """

    def _set_parameters(self, **kwargs):
        r"""Set the parameters/arguments of the factory."""
        self._entities = {"SimpleImputer": SimpleImputer}
