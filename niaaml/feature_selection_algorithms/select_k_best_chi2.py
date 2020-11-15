from niaaml.feature_selection_algorithms.feature_selection_algorithm import FeatureSelectionAlgorithm
from sklearn.feature_selection import SelectKBest, chi2

__all__ = [
	'SelectKBestChi2'
]

class SelectKBestChi2(FeatureSelectionAlgorithm):
	r"""Implementation of feature selection using selection of k best features according to the chi2 resulting data.
	
	Date:
		2020

	Author
		Luka Pečnik

	License:
		MIT
	"""

	def __init__(self, **kwargs):
		r"""Initialize the SelectKBestChi2 algorithm.
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
		k = 10
		if len(x[0]) < k:
			k = len(x[0])
		
		return SelectKBest(chi2, k=k).fit_transform(x, y)
