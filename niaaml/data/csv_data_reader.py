import pandas as pd
from niaaml.data.data_reader import DataReader

__all__ = ["CSVDataReader"]


class CSVDataReader(DataReader):
    r"""Implementation of CSV data reader.

    Date:
        2020

    Author:
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

    def _set_parameters(self, src, contains_classes=True, has_header=False, **kwargs):
        r"""Set the parameters of the algorithm.

        Arguments:
            src (string): Path to a CSV dataset file.
            contains_classes (Optional[bool]): Tells if src contains expected classification results or only features.
            has_header (Optional[bool]): Tells if src contains header row.
        """
        self.__src = src
        self.__contains_classes = contains_classes
        self.__has_header = has_header
        self._read_data()

    def _read_data(self, **kwargs):
        r"""Read data from expected source."""
        data = pd.read_csv(
            self.__src, header=None if self.__has_header is False else "infer"
        )
        header = data.columns

        if self.__contains_classes:
            self._y = data.pop(header[len(header) - 1])

        self._x = data
