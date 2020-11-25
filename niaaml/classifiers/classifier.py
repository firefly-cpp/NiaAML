
from niaaml.pipeline_component import PipelineComponent

__all__ = [
    'Classifier'
]

class Classifier(PipelineComponent):
    r"""Class for implementing classifiers.
    
    Date:
        2020

    Author
        Luka Pečnik

    License:
        MIT
    
    See Also:
        * :class:`niaaml.pipeline_component.PipelineComponent`
    """
    
    def fit(self, x, y, **kwargs):
        r"""Fit implemented classifier.

        Arguments:
            x (Iterable[any]): n samples to classify.
            y (Iterable[any]): n classes of the samples in the x array.

        Returns:
            None
        """
        return

    def predict(self, x, **kwargs):
        r"""Predict class for each sample (row) in x.

        Arguments:
            x (Iterable[any]): n samples to classify.

        Returns:
            Iterable[any]: n predicted classes.
        """
        return