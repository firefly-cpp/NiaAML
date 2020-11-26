__all__ = [
    'DataReader'
]

class DataReader:
    r"""Class for implementing data readers with different sources of data.
    
    Date:
        2020

    Author
        Luka Pečnik

    License:
        MIT

    Attributes:
        _x (numpy.ndarray[float]): Array of rows from dataset without expected classification results.
        _y (Optional[Iterable[any]]): Array of encoded expected classification results.
    """

    def __init__(self, **kwargs):
        r"""Initialize data reader.
        """
        self._x = None
        self._y = None
        self._set_parameters(**kwargs)
    
    def _set_parameters(self, **kwargs):
        r"""Set the parameters/arguments of the algorithm.
        """
        return
    
    def get_x(self, **kwargs):
        r"""Get value of _x.

        Returns:
            numpy.ndarray[float]: Array of rows from dataset without expected classification results.
        """
        return self._x
    
    def get_y(self, **kwargs):
        r"""Get value of _y.

        Returns:
            Iterable[any]: Array of encoded expected classification results.
        """
        return self._y

    def _read_data(self, **kwargs):
        r"""Read data from expected source.
        """
        return