from niaaml.utilities import Factory
from niaaml.preprocessing.feature_selection.bat_algorithm import BatAlgorithm
from niaaml.preprocessing.feature_selection.differential_evolution import (
    DifferentialEvolution,
)
from niaaml.preprocessing.feature_selection.grey_wolf_optimizer import GreyWolfOptimizer
from niaaml.preprocessing.feature_selection.jDEFSTH import jDEFSTH
from niaaml.preprocessing.feature_selection.particle_swarm_optimization import (
    ParticleSwarmOptimization,
)
from niaaml.preprocessing.feature_selection.select_k_best import SelectKBest
from niaaml.preprocessing.feature_selection.select_percentile import SelectPercentile
from niaaml.preprocessing.feature_selection.variance_threshold import VarianceThreshold

__all__ = ["FeatureSelectionAlgorithmFactory"]


class FeatureSelectionAlgorithmFactory(Factory):
    r"""Class with string mappings to feature selection algorithms.

    Attributes:
        _entities (Dict[str, FeatureSelectionAlgorithm]): Mapping from strings to feature selection algorithms.

    See Also:
        * :class:`niaaml.utilities.Factory`
    """

    def _set_parameters(self, **kwargs):
        r"""Set the parameters/arguments of the factory."""
        self._entities = {
            "jDEFSTH": jDEFSTH,
            "SelectKBest": SelectKBest,
            "SelectPercentile": SelectPercentile,
            "VarianceThreshold": VarianceThreshold,
            "BatAlgorithm": BatAlgorithm,
            "DifferentialEvolution": DifferentialEvolution,
            "GreyWolfOptimizer": GreyWolfOptimizer,
            "ParticleSwarmOptimization": ParticleSwarmOptimization,
        }
