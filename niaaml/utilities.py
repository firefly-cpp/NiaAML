from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, cohen_kappa_score, f1_score
import numpy as np

__all__ = [
	'MinMax',
	'ParameterDefinition',
	'OptimizationStats',
	'Factory',
	'float_converter',
	'get_bin_index'
]

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

def get_bin_index(value, number_of_bins):
	"""Gets index of value's bin. Value must be between 0.0 and 1.0.

	Arguments:
		value (float): Value to put into bin.
		number_of_bins (uint): Number of bins on the interval [0.0, 1.0].
	
	Returns:
		uint: Calculated index.
	"""
	bin_index = np.int(np.floor(value / (1.0 / number_of_bins)))
	if bin_index >= number_of_bins:
		bin_index -= 1
	return bin_index

class MinMax:
	r"""Class for ParameterDefinition's value property.

	Attributes:
		min (float): Minimum number (inclusive).
		max (float): Maximum number (exclusive).

	See Also:
		* :class:`niaaml.utilities.ParameterDefinition`
	"""

	def __init__(self, min, max):
		r"""Initialize instance.

		Arguments:
			min (float): Minimum number (inclusive).
			max (float): Maximum number (exclusive).
		"""
		self.min = min
		self.max = max

class ParameterDefinition:
	r"""Class for PipelineComponent parameters definition.

	Attributes:
		value (any): Array of possible parameter values or instance of MinMax class.
		param_type (numpy.dtype): Selection output data type.


	See Also:
		* :class:`niaaml.pipeline_component.PipelineComponent`
		* :class:`niaaml.utilities.MinMax`
	"""

	def __init__(self, value, param_type = None):
		r"""Initialize instance.

		Arguments:
			value (Any): Array of any type or instance of MinMax class.
			param_type (type): Type of possible outcome according to an instance of MinMax class. Not used if the argument value is array.
		"""
		self.value = value
		self.param_type = param_type

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
			return self._entities[name]()
		else:
			raise TypeError('Passed entity is not defined! --> %s' % name)

class OptimizationStats:
	r"""Class that holds pipeline optimization result's statistics. Includes accuracy, precision, Cohen's kappa and F1-score.

	Attributes:
		_accuracy (float): Calculated accuracy.
		_precision (float): Calculated precision.
		_cohen_kappa (float): Calculated Cohen's kappa.
		_f1_score (float): Calculated F1-score.
	"""

	def __init__(self, predicted, expected, **kwargs):
		r"""Initialize the factory."""
		self._accuracy = accuracy_score(expected, predicted)
		self._precision = precision_score(expected, predicted)
		self._cohen_kappa = cohen_kappa_score(expected, predicted)
		self._f1_score = f1_score(expected, predicted)
