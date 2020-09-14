from sklearn.preprocessing import StandardScaler as StdScaler
from niaaml.preprocessing_algorithms.preprocessing_algorithm import PreprocessingAlgorithm

__all__ = ['StandardScaler']

class StandardScaler(PreprocessingAlgorithm):
    r"""Implementation of feature standard scaling algorithm.
    
	Date:
		2020

	Author
		Luka Pečnik

	License:
        MIT

	See Also:
		* :class:`niaaml.preprocessing_algorithms.PreprocessingAlgorithm`
    """

    def process(self, x, **kwargs):
        r"""Processes the given x data.

        Arguments:
            x (numpy.ndarray[float]): Data to process.

        Returns:
            numpy.ndarray[float]: Processed data.
        
        See Also:
            * :func:`niaaml.preprocessing_algorithms.PreprocessingAlgorithm.process`
        """
        
        return StdScaler().fit_transform(x)
