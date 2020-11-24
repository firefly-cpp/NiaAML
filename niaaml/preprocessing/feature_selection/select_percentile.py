from niaaml.utilities import ParameterDefinition, MinMax
from niaaml.preprocessing.feature_selection.feature_selection_algorithm import FeatureSelectionAlgorithm
from sklearn.feature_selection import SelectPercentile, chi2
import numpy as np

__all__ = [
	'SelectPercentileFeatureSelection'
]

class SelectPercentileFeatureSelection(FeatureSelectionAlgorithm):
	r"""Implementation of feature selection using percentile selection of best features according to used score function.
	
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
		r"""Initialize SelectPercentile feature selection algorithm.
		"""
		self._params = dict(
			score_func = ParameterDefinition([chi2]),
			percentile = ParameterDefinition(MinMax(10, 100), np.uint)
		)
		self.__select_percentile = SelectPercentile()

	def set_parameters(self, **kwargs):
		r"""Set the parameters/arguments of the algorithm.
		"""
		self.__select_percentile.set_params(**kwargs)
	
	def select_features(self, x, y, **kwargs):
		r"""Perform the feature selection process.

		Arguments:
			x (Iterable[any]): Array of original features.
            y (Iterable[int]): Array of expected classes (ignored, but available for compatibility with other feature selection algorithms).

		Returns:
			Iterable[bool]: Mask of selected features.
		"""
		self.__select_percentile.fit(x, y)
		return self.__select_percentile.get_support()
