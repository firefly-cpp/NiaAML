from sklearn import preprocessing
import numpy

__all__ = [
	'MinMax',
	'ParameterDefinition',
	'Factory',
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
	r"""TODO
	"""
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
	r"""TODO
	"""
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

class Factory:
	r"""Base class with string mappings to entities.

	Attributes:
		_entities (Dict[str, any]): Dictionary to map from strings to an instance of anything.
	"""
	_entities = None

	def __init__(self, **kwargs):
		r"""Initialize the factory."""
		self._set_parameters(**kwargs)
	
	def _set_parameters(self, **kwargs):
		r"""Set the parameters/arguments of the factory.
		"""
		return
	
	def get_result(self, name):
		r"""Get the resulting entity.

		Arguments:
			name (str): String that represents the entity.

		Returns:
			any: Entity according to the given name.
		"""

		if name in self._entities:
			return self._entities[name]
		else:
			raise TypeError('Passed entity is not defined! --> %s' % name) 