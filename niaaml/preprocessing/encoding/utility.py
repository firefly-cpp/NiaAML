from niaaml.utilities import Factory
import pandas as pd
from niaaml.preprocessing.encoding.one_hot_encoder import OneHotEncoder

__all__ = ["encode_categorical_features", "EncoderFactory"]


def encode_categorical_features(features, encoder):
    """Encode categorical features.

    Arguments:
        features (pandas.core.frame.DataFrame): DataFrame of features.
        encoder (str): Name of the encoder to use.

    Returns:
                Tuple[pandas.core.frame.DataFrame, Iterable[FeatureEncoder]]:
                        1. Converted dataframe.
                        2. Dictionary of encoders for all categorical features.
    """
    enc = EncoderFactory().get_result(encoder)

    encoders = {}
    to_drop = []
    enc_features = pd.DataFrame()
    cols = [
        col
        for col in features.columns
        if not pd.api.types.is_numeric_dtype(features[col])
    ]
    for c in cols:
        enc.fit(features[[c]])
        tr = enc.transform(features[[c]])
        to_drop.append(c)
        enc_features = pd.concat([enc_features, tr], axis=1)
        encoders[c] = enc
    features = features.drop(to_drop, axis=1)
    features = pd.concat([features, enc_features], axis=1)
    return features, encoders if len(encoders) > 0 else None


class EncoderFactory(Factory):
    r"""Class with string mappings to encoders.

    Attributes:
        _entities (Dict[str, FeatureEncoder]): Mapping from strings to encoders.

    See Also:
        * :class:`niaaml.utilities.Factory`
    """

    def _set_parameters(self, **kwargs):
        r"""Set the parameters/arguments of the factory."""
        self._entities = {"OneHotEncoder": OneHotEncoder}
