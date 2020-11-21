import numpy as np
from niaaml.utilities import MinMax

__all__ = [
	'PipelineComponent'
]

class PipelineComponent:
	r"""Class for implementing pipeline components.
	
	Date:
		2020

	Author
		Luka Pečnik

	License:
		MIT

	Attributes:
		_params (Dict[str, ParameterDefinition]): Dictionary of components's parameters with possible values. Possible parameter values are given as an instance of the ParameterDefinition class.
	
	See Also:
		* :class:`niaaml.utilities.ParameterDefinition`
    """

	def __init__(self, **kwargs):
		r"""Initialize pipeline component.
		"""
		# _params variable should not be static as in some cases it is instance specific
		self._params = None
		self._set_parameters(**kwargs)
	
	def _set_parameters(self, **kwargs):
		r"""Set the parameters/arguments of the pipeline component.
		"""
		return
	 
	@classmethod
	def getRandomInstance(i):
		r"""Randomly initialize instance of the implemented `niaaml.pipeline_component.PipelineComponent` class.

        Arguments:
            i (PipelineComponent): Any class that implements PipelineComponent class.

        Returns:
            PipelineComponent: Randomly initialized PipelineComponent instance.
		"""
		instance = i()
		params = dict()

		if instance._params:
			for key, value in instance._params.items():
				# value should be somehow determined runtime in case its value is currently None and added to the _params dictionary to include it into the optimization process
				if value is not None:
					if isinstance(value.value, MinMax):
						val = np.random.uniform(value.value.min, value.value.max)
						if value.param_type is np.intc or value.param_type is np.int or value.param_type is np.uintc or value.param_type is np.uint:
							val = value.param_type(np.around(val))
							if val >= value.value.max:
								val = value.value.max - 1
						params[key] = val
					else:
						params[key] = value.value[np.random.randint(0, len(value.value))]
			
			instance._set_parameters(**params)
		
		return instance
