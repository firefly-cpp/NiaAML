from sklearn.metrics import cohen_kappa_score

__all__ = [
    'CohenKappa'
]

class CohenKappa:
    r"""Class representing the cohen's kappa as a fitness function.
    
    Date:
        2020

    Author
        Luka Pečnik

    License:
        MIT
    """
    
    def get_fitness(self, predicted, expected):
        r"""Return fitness value. The larger return value should represent a better fitness for the framework to work properly.

        Arguments:
            predicted (Iterable[any]): Predicted values.
            expected (Iterable[any]): Expected values.
        
        Returns:
            float: Calculated fitness value.
        """
        return cohen_kappa_score(expected, predicted)
