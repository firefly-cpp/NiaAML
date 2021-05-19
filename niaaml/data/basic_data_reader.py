import pandas as pd
from niaaml.data.data_reader import DataReader

__all__ = ["BasicDataReader"]


class BasicDataReader(DataReader):
    r"""Implementation of basic data reader.

    Date:
        2020

    Author:
        Luka Pečnik

    License:
        MIT

    See Also:
        * :class:`niaaml.data.DataReader`
    """

    def _set_parameters(self, x, y=None, **kwargs):
        r"""Set the parameters of the algorithm.

        Arguments:
            x (Iterable[float]): Array of rows from dataset without expected classification results.
            y (Optional[Iterable[any]]): Array of expected classification results.
        """
        self._x = pd.DataFrame(x)

        if y is not None:
            self._y = pd.Series(y)
