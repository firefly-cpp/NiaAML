from niaaml.utilities import Factory
from niaaml import feature_selection_algorithms

__all__ = [
	'FeatureSelectionAlgorithmFactory'
]

class FeatureSelectionAlgorithmFactory(Factory):
	r"""Class with string mappings to feature selection algorithms.

	Attributes:
		_entities (Dict[str, FeatureSelectionAlgorithm]): Mapping from strings to feature selection algorithms.
	"""

	def _set_parameters(self, **kwargs):
		r"""Set the parameters/arguments of the factory.

		See Also:
			* :func:`niaaml.utilities.Factory._set_parameters`
		"""
		self._entities = {
			'jDEFSTH': feature_selection_algorithms.jDEFSTH,
			'SelectKBestChi2': feature_selection_algorithms.SelectKBestChi2,
			'SelectPercentileChi2': feature_selection_algorithms.SelectPercentileChi2,
			'VarianceThresholdFeatureSelection': feature_selection_algorithms.VarianceThresholdFeatureSelection
		}