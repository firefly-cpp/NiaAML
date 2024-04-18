__all__ = ["FitnessFunction"]


class FitnessFunction:
    r"""Class for implementing fitness functions.

    Date:
        2020

    Author:
        Luka Pečnik

    License:
        MIT

    Attributes:
        Name (str): Name of the fitness function.
    """
    Name = None

    def __init__(self, **kwargs):
        r"""Initialize fitness function."""
        self.set_parameters(**kwargs)

    def set_parameters(self, **kwargs):
        r"""Set the parameters/arguments of the pipeline component."""
        return

    def get_fitness(self, predicted, expected):
        r"""Return fitness value. The larger return value should represent a better fitness for the framework to work properly.

        Arguments:
            predicted (pandas.core.series.Series): Predicted values.
            expected (pandas.core.series.Series): Expected values.

        Returns:
            float: Calculated fitness value.
        """
        return None
    
    def get_bounds(self):
        """Returns the optimization bounds for this fitness function.
        
        The default is for classification metrics.
        
        Retunrs:
            Tuple[float, float]: lower and upper optimization bounds. Defaults to (0.0, 1.0)"""
        return (0.0, 1.0)
