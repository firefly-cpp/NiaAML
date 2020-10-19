from niaaml.feature_selection_algorithms.feature_selection_algorithm import FeatureSelectionAlgorithm
from sklearn.feature_selection import VarianceThreshold

__all__ = [
	'VarianceThresholdFeatureSelection'
]

class VarianceThresholdFeatureSelection(FeatureSelectionAlgorithm):
	r"""Implementation of feature selection using variance threshold.
	
	Date:
		2020

	Author
		Luka Pečnik

	License:
		MIT
	"""

	def __init__(self, **kwargs):
		r"""Initialize the VarianceThresholdFeatureSelection algorithm.
		"""
		self._set_parameters(**kwargs)
	
	def _set_parameters(self, **kwargs):
		r"""Set the parameters/arguments of the algorithm.
		"""
		return
	
	def select_features(self, x, y, **kwargs):
		r"""Perform the feature selection process.

		Arguments:
			x (Iterable[any]): Array of original features.
            y (Iterable[int]): Array of expected classes (ignored, but available for compatibility with other feature selection algorithms).

		Returns:
			Iterable[any]: Array of selected features.
		"""
		return VarianceThreshold().fit_transform(x, )
