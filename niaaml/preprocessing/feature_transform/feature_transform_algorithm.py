from niaaml.preprocessing.preprocessing_algorithm import PreprocessingAlgorithm

__all__ = [
    'FeatureTransformAlgorithm'
]

class FeatureTransformAlgorithm(PreprocessingAlgorithm):
    r"""Class for implementing feature transform algorithms.
    
    Date:
        2020

    Author
        Luka Pečnik

    License:
        MIT

    See Also:
        * :class:`niaaml.preprocessing.preprocessing_algorithm.PreprocessingAlgorithm`
    """

    def fit(self, x, **kwargs):
        r"""Fit implemented classifier.

        Arguments:
            x (Iterable[any]): n samples to fit transformation algorithm.
        """
        return
    
    def transform(self, x, **kwargs):
        r"""Transforms the given x data.

        Arguments:
            x (Iterable[any]): Data to transform.

        Returns:
            Iterable[any]: Transformed data.
        """
        return x
