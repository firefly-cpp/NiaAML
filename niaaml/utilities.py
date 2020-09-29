from sklearn import preprocessing
import numpy

__all__ = [
	'MinMax',
	'ParameterDefinition',
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

	def __init__(self, min, max):
		r"""Initialize instance.
		"""
		self.min = min
		self.max = max

class ParameterDefinition:
	value = None
	paramType = None

	def __init__(self, value, paramType):
		r"""Initialize instance.
		"""
		self.value = value
		self.paramType = paramType