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
		self.set_parameters(**kwargs)
	
	def set_parameters(self, **kwargs):
		r"""Set the parameters/arguments of the pipeline component.
		"""
		return
	
	def get_params_dict_size(self):
		r"""Return parameters definition dictionary.
		"""
		return self._params
