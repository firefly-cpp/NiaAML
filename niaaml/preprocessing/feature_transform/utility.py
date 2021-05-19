from niaaml.utilities import Factory
from niaaml.preprocessing.feature_transform.normalizer import Normalizer
from niaaml.preprocessing.feature_transform.standard_scaler import StandardScaler
from niaaml.preprocessing.feature_transform.max_abs_scaler import MaxAbsScaler
from niaaml.preprocessing.feature_transform.quantile_transformer import (
    QuantileTransformer,
)
from niaaml.preprocessing.feature_transform.robust_scaler import RobustScaler

__all__ = ["FeatureTransformAlgorithmFactory"]


class FeatureTransformAlgorithmFactory(Factory):
    r"""Class with string mappings to feature transform algorithms.

    Attributes:
        _entities (Dict[str, FeatureTransformAlgorithm]): Mapping from strings to feature transform algorithms.
    """

    def _set_parameters(self, **kwargs):
        r"""Set the parameters/arguments of the factory."""
        self._entities = {
            "Normalizer": Normalizer,
            "StandardScaler": StandardScaler,
            "MaxAbsScaler": MaxAbsScaler,
            "QuantileTransformer": QuantileTransformer,
            "RobustScaler": RobustScaler,
        }
