from niaaml.utilities import ParameterDefinition, MinMax
from niaaml.preprocessing.feature_selection.feature_selection_algorithm import FeatureSelectionAlgorithm
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np

__all__ = [
	'SelectKBestFeatureSelection'
]

class SelectKBestFeatureSelection(FeatureSelectionAlgorithm):
	r"""Implementation of feature selection using selection of k best features according to used score function.
	
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
		r"""Initialize SelectKBest feature selection algorithm.

		Notes:
			_params['k'] is initialized to None as it is included in the optimization process later since we cannot determine a proper value range until length of the feature vector becomes known.
		"""
		self._params = dict(
			score_func = ParameterDefinition([chi2]),
			k = None
		)
		self.__k = None
		self.__select_k_best = SelectKBest()
	
	def set_parameters(self, **kwargs):
		r"""Set the parameters/arguments of the algorithm.
		"""
		self.__select_k_best.set_params(**kwargs)
	
	def select_features(self, x, y, **kwargs):
		r"""Perform the feature selection process.

		Arguments:
			x (Iterable[any]): Array of original features.
            y (Iterable[any]): Array of expected classes (ignored, but available for compatibility with other feature selection algorithms).

		Returns:
			Iterable[bool]: Mask of selected features.
		"""
		if self.__k is None:
			self.__k = len(x[0])
			self._params['k'] = ParameterDefinition(MinMax(1, self.__k), np.int)
			val = np.int(np.around(np.random.uniform(1, self.__k)))
			self.__select_k_best.set_params(k=val)
		
		self.__select_k_best.fit(x, y)
		return self.__select_k_best.get_support()
