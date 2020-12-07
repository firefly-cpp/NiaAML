import niaaml.preprocessing.encoding as enc
from niaaml.utilities import Factory
import pandas as pd
import numpy as np

__all__ = [
    'encode_categorical_features',
    'EncoderFactory'
]

def encode_categorical_features(features, encoder):
    """Encode categorical features.

    Arguments:
        features (pandas.core.frame.DataFrame): DataFrame of features.
        encoder (str): Number of bins on the interval [0.0, 1.0].
    
    Returns:
		Iterable[FeatureEncoder]: Encoder for each categorical feature encoded.
		Tuple[pandas.core.frame.DataFrame, Iterable[FeatureEncoder]]:
			1. Converted dataframe.
			2. List of encoders for all categorical features.
    """
    enc = EncoderFactory().get_result(encoder)

    encoders = []
    types = features.dtypes
    to_drop = []
    enc_features = pd.DataFrame()
    for i in range(len(types)):
        if types[i] != np.dtype('float64'):
            enc.fit(features[[i]])
            tr = enc.transform(features[[i]])
            to_drop.append(i)
            enc_features = pd.concat([enc_features, tr], axis=1)
            encoders.append(enc)
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
        r"""Set the parameters/arguments of the factory.
        """
        self._entities = {
            'OneHotEncoder': enc.OneHotEncoder
        }
