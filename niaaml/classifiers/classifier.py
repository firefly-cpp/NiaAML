__all__ = [
	'Classifier'
]

class Classifier:
	r"""Class for implementing classifiers.
	
	Date:
		2020

	Author
		Luka Pečnik

	License:
		MIT

	Attributes:
		_params (Dict[str, Any]): Dictionary of classifier's parameters with possible values. Possible parameter values are either given as an array of values (categoric parameters) or an instance of a `niaaml.utilities.MinMax` class (numeric parameters).
	
	See Also:
		* :class:`niaaml.utilities.MinMax`
    """

	_params = None

	def __init__(self, **kwargs):
		r"""Initialize data reader.
		"""
		self._set_parameters(**kwargs)
	
	def _set_parameters(self, **kwargs):
		r"""Set the parameters/arguments of the algorithm.
		"""
		return
	
	def fit(self, x, y, **kwargs):
		r"""Fit implemented classifier.
		"""
		return

	def predict(self, x, **kwargs):
		r"""Predict class for each sample (row) in x.

        Arguments:
            x (numpy.ndarray[float]): n samples to classify.

        Returns:
            numpy.array[int]: n predicted classes.
		"""
		return
