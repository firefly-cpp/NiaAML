from niaaml.preprocessing.preprocessing_algorithm import PreprocessingAlgorithm

__all__ = ["FeatureSelectionAlgorithm"]


class FeatureSelectionAlgorithm(PreprocessingAlgorithm):
    r"""Class for implementing feature selection algorithms.

    Date:
        2020

    Author:
        Luka Pečnik

    License:
        MIT

    See Also:
        * :class:`niaaml.preprocessing.preprocessing_algorithm.PreprocessingAlgorithm`
    """

    def select_features(self, x, y, **kwargs):
        r"""Perform the feature selection process.

        Arguments:
            x (pandas.core.frame.DataFrame): Array of original features.
            y (pandas.core.series.Series) Expected classifier results.

        Returns:
            numpy.ndarray[bool]: Mask of selected features.
        """
        return x
