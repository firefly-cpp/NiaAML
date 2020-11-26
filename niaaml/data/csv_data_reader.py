import csv
import numpy as np
from niaaml.data.data_reader import DataReader

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
		__src (string): Path to a CSV file.
		__contains_classes (bool): Tells if src contains expected classification results or only features.
		__has_header (bool): Tells if src contains header row.

	See Also:
		* :class:`niaaml.data.DataReader`
	"""

	def _set_parameters(self, src, contains_classes = True, has_header = False, **kwargs):
		r"""Set the parameters of the algorithm.

		Arguments:
			src (string): Path to a CSV dataset file.
			contains_classes (bool): Tells if src contains expected classification results or only features.
			has_header (bool): Tells if src contains header row.
		"""
		self.__src = src
		self.__contains_classes = contains_classes
		self.__has_header = has_header
		self._read_data()

	def _read_data(self, **kwargs):
		r"""Read data from expected source.
		"""
		self._x = []

		if self.__contains_classes:
			self._y = []

		with open(self.__src) as csvfile:
			reader = csv.reader(csvfile)

			if self.__has_header:
				next(reader, None)

			y = []
			for row in reader:
				if self.__contains_classes:
					self._x.append(np.array(row[:-1], dtype=np.float))
					y.append(row[-1])
				else:
					self._x.append(np.array(row, dtype=np.float))

			if self.__contains_classes:
				self._y = y