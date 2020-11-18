from niaaml.utilities import Factory
import niaaml.preprocessing.feature_transform as ft

__all__ = [
	'FeatureTransformAlgorithmFactory'
]

class FeatureTransformAlgorithmFactory(Factory):
	r"""Class with string mappings to feature transform algorithms.

	Attributes:
		_entities (Dict[str, FeatureTransformAlgorithm]): Mapping from strings to feature transform algorithms.
	"""

	def _set_parameters(self, **kwargs):
		r"""Set the parameters/arguments of the factory.

		See Also:
			* :func:`niaaml.utilities.Factory._set_parameters`
		"""
		self._entities = {
			'Normalizer': ft.Normalizer,
			'StandardScaler': ft.StandardScaler
		}
