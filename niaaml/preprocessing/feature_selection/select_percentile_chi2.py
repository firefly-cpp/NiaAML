from niaaml.preprocessing.feature_selection.feature_selection_algorithm import FeatureSelectionAlgorithm
from sklearn.feature_selection import SelectPercentile, chi2

__all__ = [
	'SelectPercentileChi2'
]

class SelectPercentileChi2(FeatureSelectionAlgorithm):
	r"""Implementation of feature selection using percentile selection of best features according to the chi2 resulting data.
	
	Date:
		2020

	Author
		Luka Pečnik

	License:
		MIT
	"""

	def __init__(self, **kwargs):
		r"""Initialize the SelectPercentileChi2 algorithm.
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
		return SelectPercentile(chi2).fit_transform(x, y)
