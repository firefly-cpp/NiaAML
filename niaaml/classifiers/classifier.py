from niaaml.pipeline_component import PipelineComponent

__all__ = ["Classifier"]


class Classifier(PipelineComponent):
    r"""Class for implementing classifiers.

    Date:
        2020

    Author:
        Luka Pečnik

    License:
        MIT

    See Also:
        * :class:`niaaml.pipeline_component.PipelineComponent`
    """

    def fit(self, x, y, **kwargs):
        r"""Fit implemented classifier.

        Arguments:
            x (pandas.core.frame.DataFrame): n samples to classify.
            y (pandas.core.series.Series): n classes of the samples in the x array.
        """
        return

    def predict(self, x, **kwargs):
        r"""Predict class for each sample (row) in x.

        Arguments:
            x (pandas.core.frame.DataFrame): n samples to classify.

        Returns:
            pandas.core.series.Series: n predicted classes.
        """
        return
