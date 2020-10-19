import csv
import numpy as np
from niaaml.data.data_reader import DataReader
from niaaml.utilities import get_label_encoder, float_converter

__all__ = ['CSVDataReader']

class CSVDataReader(DataReader):
	r"""Implementation of CSV data reader.
	
	Date:
		2020

	Author
		Luka Pečnik

	License:
		MIT

	Attributes:
		_src (string): Path to a CSV file.

	See Also:
		* :class:`niaaml.data.DataReader`
	"""

	def _set_parameters(self, src, **kwargs):
		r"""Set the parameters of the algorithm.

		Arguments:
			src (string): Path to a CSV dataset file.

		See Also:
			* :func:`niaaml.data.DataReader._set_parameters`
		"""
		self.__src = src
		self._read_data()

	def _read_data(self, **kwargs):
		r"""Read data from expected source.

		See Also:
			* :func:`niaaml.data.DataReader._read_data`
		"""
		self._x = []
		y = []
		with open(self.__src) as csvfile:
			reader = csv.reader(csvfile)
			next(reader, None)
			for row in reader:
				self._x.append(float_converter(row[1:]))
				y.append(row[0])

			self._label_encoder = get_label_encoder(y)
			self._y = np.array(self._label_encoder.transform(y)).astype(np.uintc)