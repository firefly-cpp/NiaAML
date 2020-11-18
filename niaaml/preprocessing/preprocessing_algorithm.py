__all__ = [
	'PreprocessingAlgorithm'
]

class PreprocessingAlgorithm:
	r"""Class for implementing preprocessing algorithms.
	
	Date:
		2020

	Author
		Luka Pečnik

	License:
		MIT
	"""

	def __init__(self, **kwargs):
		r"""Initialize preprocessing algorithm.
		"""
		self._set_parameters(**kwargs)
	
	def _set_parameters(self, **kwargs):
		r"""Set the parameters/arguments of the algorithm.
		"""
		return

	@classmethod
	def getRandomInstance(pa):
		r"""Randomly initialize instance of the `niaaml.preprocessing.PreprocessingAlgorithm` class.

		Arguments:
			pa (PreprocessingAlgorithm): Any class that implements PreprocessingAlgorithm class.

		Returns:
			PreprocessingAlgorithm: Randomly initialized PreprocessingAlgorithm instance.
		"""
		instance = pa()
		return instance
		"""
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
		"""
