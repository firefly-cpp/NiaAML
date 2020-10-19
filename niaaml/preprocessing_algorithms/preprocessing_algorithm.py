__all__ = [
    'PreprocessingAlgorithm'
]

class PreprocessingAlgorithm:
    r"""Class for implementing preprocessing algorithms.
    
	Date:
		2020

	Author
		Luka Pečnik

	License:
        MIT
    """

    def __init__(self, **kwargs):
        r"""Initialize preprocessing algorithm.
        """
        self._set_parameters(**kwargs)
    
    def _set_parameters(self, **kwargs):
        r"""Set the parameters/arguments of the algorithm.
        """
        return
    
    def process(self, x, **kwargs):
        r"""Processes the given x data.

        Arguments:
            x (Iterable[any]): Data to process.

        Returns:
            Iterable[any]: Processed data.
        """
        return x
