from sklearn import preprocessing
import numpy

__all__ = [
	'MinMax',
	'get_label_encoder'
]

def get_label_encoder(labels):
	r"""Get the label encoder instance.

	Arguments:
		labels (Iterable[string]): Array of labels.

	Returns:
		sklearn.preprocessing.LabelEncoder: Instance of a LabelEncoder fit to the given labels.
	"""

	le = preprocessing.LabelEncoder()
	le.fit(labels)
	return le

class MinMax:
	min = 0
	max = 0

	def __init__(self, **kwargs):
		r"""Initialize instance.
		"""
		self._set_parameters(**kwargs)
	
	def _set_parameters(self, min, max, **kwargs):
		r"""Set the parameters of the algorithm.

		Arguments:
			min (numpy.double): Lower bound.
			max (numpy.double): Upper bound.
		"""
		self.min = min
		self.max = max