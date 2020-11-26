from sklearn.metrics import accuracy_score

__all__ = [
    'Accuracy'
]

class Accuracy:
    r"""Class representing the accuracy as a fitness function.
    
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
        return accuracy_score(expected, predicted)
