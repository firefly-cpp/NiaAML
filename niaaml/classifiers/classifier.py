from niaaml.utilities import MinMax
import numpy as np

__all__ = [
	'Classifier'
]

class Classifier:
	r"""Class for implementing classifiers.
	
	Date:
		2020

	Author
		Luka Pečnik

	License:
		MIT

	Attributes:
		_params (Dict[str, ParameterDefinition]): Dictionary of classifier's parameters with possible values. Possible parameter values are given as an instance of the ParameterDefinition class.
	
	See Also:
		* :class:`niaaml.utilities.ParameterDefinition`
    """

	_params = None

	def __init__(self, **kwargs):
		r"""Initialize data reader.
		"""
		self._set_parameters(**kwargs)
	
	def _set_parameters(self, **kwargs):
		r"""Set the parameters/arguments of the algorithm.
		"""
		return
	
	def fit(self, x, y, **kwargs):
		r"""Fit implemented classifier.

        Arguments:
            x (Iterable[any]): n samples to classify.
			y (numpy.array[int]): n classes of the samples in the x array.

        Returns:
            None
		"""
		return

	def predict(self, x, **kwargs):
		r"""Predict class for each sample (row) in x.

        Arguments:
            x (Iterable[any]): n samples to classify.

        Returns:
            numpy.array[int]: n predicted classes.
		"""
		return
	 
	@classmethod
	def getRandomInstance(cls):
		r"""Randomly initialize instance of the `niaaml.classifiers.Classifier` class.

        Arguments:
            cls (Classifier): Any class that implements Classifier class.

        Returns:
            Classifier: Randomly initialized Classifier instance.
		"""
		instance = cls()
		params = dict()

		if cls._params:
			for key, value in cls._params.items():
				if isinstance(value.value, MinMax):
					val = np.random.uniform(value.value.min, value.value.max)
					if value.paramType is np.intc or value.paramType is np.uintc or value.paramType is np.uint:
						val = value.paramType(np.around(val))
					params[key] = val
				else:
					params[key] = value.value[np.random.randint(0, len(value.value))]
			
			instance._set_parameters(**params)
			return instance
		
		return None
