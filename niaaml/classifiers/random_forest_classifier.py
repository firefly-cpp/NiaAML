from niaaml.classifiers.classifier import Classifier
from niaaml.utilities import MinMax
from sklearn.ensemble import RandomForestClassifier

__all__ = ['RandomForestClassifier']

class RandomForestClassifier(Classifier):
	r"""Implementation of random forest classifier.
	
	Date:
		2020

	Author
		Luka Pečnik

	License:
		MIT

	See Also:
		* :class:`niaaml.classifiers.Classifier`
	"""
	__randomForestClassifier = RandomForestClassifier()

	def _set_parameters(self, **kwargs):
		r"""Set the parameters/arguments of the algorithm.

		See Also:
			* :func:`niaaml.classifiers.Classifier._set_parameters`
		"""
		Classifier._set_parameters(self, **kwargs)
		self._params = dict(
			n_estimators = MinMax(min=10, max=150)
		)

		#self.__randomForestClassifier.set_params(**{"n_estimators": 15})
