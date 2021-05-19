from niaaml.preprocessing.preprocessing_algorithm import PreprocessingAlgorithm

__all__ = ["FeatureTransformAlgorithm"]


class FeatureTransformAlgorithm(PreprocessingAlgorithm):
    r"""Class for implementing feature transform algorithms.

    Date:
        2020

    Author:
        Luka Pečnik

    License:
        MIT

    See Also:
        * :class:`niaaml.preprocessing.preprocessing_algorithm.PreprocessingAlgorithm`
    """

    def fit(self, x, **kwargs):
        r"""Fit implemented feature transform algorithm.

        Arguments:
            x (pandas.core.frame.DataFrame): n samples to fit transformation algorithm.
        """
        return

    def transform(self, x, **kwargs):
        r"""Transforms the given x data.

        Arguments:
            x (pandas.core.frame.DataFrame): Data to transform.

        Returns:
            pandas.core.frame.DataFrame: Transformed data.
        """
        return x
