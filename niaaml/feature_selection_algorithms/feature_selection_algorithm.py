__all__ = [
	'FeatureSelectionAlgorithm'
]

class FeatureSelectionAlgorithm:
	r"""Class for implementing feature selection algorithms.
	
	Date:
		2020

	Author
		Luka Pečnik

	License:
		MIT
	"""

	def __init__(self, **kwargs):
		r"""Initialize feature selection algorithm.
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
            y (Iterable[int]): Array of expected classes.

		Returns:
			Iterable[any]: Array of selected features.
		"""
		return x
