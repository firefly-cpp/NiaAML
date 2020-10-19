from sklearn import preprocessing
import numpy

__all__ = [
	'MinMax',
	'ParameterDefinition',
	'get_label_encoder',
	'float_converter'
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

def float_converter(array):
	r"""Convert values in the array to float if possible, leave element as is otherwise.

	Arguments:
		array (Iterable[string]): Array of strings.

	Returns:
		Iterable[any]: Array of mixed types (floats and strings).
	"""

	converted_array = []
	for element in array:
		try:
			converted_array.append(float(element))
		except ValueError:
			converted_array.append(element)
	return converted_array

class MinMax:
	min = 0
	max = 0

	def __init__(self, min, max):

		r"""Initialize instance.

		Arguments:
			min (float): Minimum number.
			max (float): Maximum number.
		"""
		self.min = min
		self.max = max

class ParameterDefinition:
	value = None
	paramType = None

	def __init__(self, value, paramType):
		r"""Initialize instance.

		Arguments:
			value (Any): Array of any type or instance of MinMax class.
			paramType (type): Type of possible outcome according to an instance of MinMax class. Not used if the argument value is array.
		"""
		self.value = value
		self.paramType = paramType