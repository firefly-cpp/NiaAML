import numpy as np
from niaaml.data.data_reader import DataReader
from niaaml.utility import get_label_encoder

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

	def _set_parameters(self, x, y, **kwargs):
		r"""Set the parameters of the algorithm.

		Arguments:
			x (Iterable[float]): Array of rows from dataset without expected classification results.
			y (Iterable[string]): Array of expected classification results.

		See Also:
			* :func:`niaaml.data.DataReader._set_parameters`
		"""
		DataReader._set_parameters(self, **kwargs)

		self._x = np.array(x).astype(np.float)

		self._label_encoder = get_label_encoder(y)
		self._y = np.array(self._label_encoder.transform(y)).astype(np.uintc)
