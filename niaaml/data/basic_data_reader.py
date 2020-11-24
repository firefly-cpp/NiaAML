import numpy as np
from niaaml.data.data_reader import DataReader
from niaaml.utilities import float_converter

__all__ = ['BasicDataReader']

class BasicDataReader(DataReader):
	r"""Implementation of basic data reader.
	
	Date:
		2020

	Author
		Luka Pečnik

	License:
		MIT

	See Also:
		* :class:`niaaml.data.DataReader`
	"""

	def _set_parameters(self, x, y=None, **kwargs):
		r"""Set the parameters of the algorithm.

		Arguments:
			x (Iterable[any]): Array of rows from dataset without expected classification results.
			y (Optional[Iterable[string]]): Array of expected classification results.

		See Also:
			* :func:`niaaml.data.DataReader._set_parameters`
		"""
		self._x = float_converter(x)
		
		if y is not None:
			self._y = y
