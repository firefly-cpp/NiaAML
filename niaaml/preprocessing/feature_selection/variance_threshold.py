from niaaml.utilities import ParameterDefinition, MinMax
from niaaml.preprocessing.feature_selection.feature_selection_algorithm import FeatureSelectionAlgorithm
from sklearn.feature_selection import VarianceThreshold
import numpy as np

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

	See Also:
		* :class:`niaaml.preprocessing.feature_selection.feature_selection_algorithm.FeatureSelectionAlgorithm`
	"""

	def __init__(self, **kwargs):
		r"""Initialize VarianceThreshold feature selection algorithm.
		"""
		self._params = dict(
			threshold = ParameterDefinition(MinMax(0, 0.1), np.float)
		)
		self.__variance_threshold = VarianceThreshold()

	def set_parameters(self, **kwargs):
		r"""Set the parameters/arguments of the algorithm.
		"""
		self.__variance_threshold.set_params(**kwargs)
	
	def select_features(self, x, y, **kwargs):
		r"""Perform the feature selection process.

		Arguments:
			x (Iterable[any]): Array of original features.
            y (Iterable[int]): Array of expected classes (ignored, but available for compatibility with other feature selection algorithms).

		Returns:
			Iterable[bool]: Mask of selected features.
		"""
		self.__variance_threshold.fit(x)
		return self.__variance_threshold.get_support()
