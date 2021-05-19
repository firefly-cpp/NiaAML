__all__ = ["DataReader"]


class DataReader:
    r"""Class for implementing data readers with different sources of data.

    Date:
        2020

    Author:
        Luka Pečnik

    License:
        MIT

    Attributes:
        _x (pandas.core.frame.DataFrame): Array of rows from dataset without expected classification results.
        _y (Optional[pandas.core.series.Series]): Array of encoded expected classification results.
    """

    def __init__(self, **kwargs):
        r"""Initialize data reader."""
        self._x = None
        self._y = None
        self._set_parameters(**kwargs)

    def _set_parameters(self, **kwargs):
        r"""Set the parameters/arguments of the algorithm."""
        return

    def get_x(self):
        r"""Get value of _x.

        Returns:
            pandas.core.frame.DataFrame: Array of rows from dataset without expected classification results.
        """
        return self._x

    def get_y(self):
        r"""Get value of _y.

        Returns:
            pandas.core.series.Series: Array of encoded expected classification results.
        """
        return self._y

    def set_x(self, value):
        r"""Set the value of _x."""
        self._x = value

    def set_y(self, value):
        r"""Set the value of _y."""
        self._y = value

    def _read_data(self):
        r"""Read data from expected source."""
        return
