from niaaml.utilities import Factory
import niaaml.preprocessing.feature_selection as fs

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
			'jDEFSTH': fs.jDEFSTH,
			'SelectKBestFeatureSelection': fs.SelectKBestFeatureSelection,
			'SelectPercentileFeatureSelection': fs.SelectPercentileFeatureSelection,
			'VarianceThresholdFeatureSelection': fs.VarianceThresholdFeatureSelection
		}
