__all__ = [
    'FitnessFunction'
]

class FitnessFunction:
    r"""Class for implementing fitness functions.
    
    Date:
        2020

    Author
        Luka Pečnik

    License:
        MIT
    """

    def __init__(self, **kwargs):
        r"""Initialize fitness function.
        """
        return
    
    def set_parameters(self, **kwargs):
        r"""Set the parameters/arguments of the pipeline component.
        """
        return
    
    def get_fitness(self, predicted, expected):
        r"""Return fitness value. The larger return value should represent a better fitness for the framework to work properly.

        Arguments:
            predicted (Iterable[any]): Predicted values.
            expected (Iterable[any]): Expected values.
        
        Returns:
            float: Calculated fitness value.
        """
        return None
