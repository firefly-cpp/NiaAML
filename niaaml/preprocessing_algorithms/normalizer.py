from sklearn.preprocessing import normalize
from niaaml.preprocessing_algorithms.preprocessing_algorithm import PreprocessingAlgorithm

__all__ = ['Normalizer']

class Normalizer(PreprocessingAlgorithm):
    r"""Implementation of feature normalization algorithm.
    
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
            x (Iterable[any]): Data to process.

        Returns:
            Iterable[any]: Processed data.
        
        See Also:
            * :func:`niaaml.preprocessing_algorithms.PreprocessingAlgorithm.process`
        """

        return normalize(x, axis=0)
