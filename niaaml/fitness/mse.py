from sklearn.metrics import mean_squared_error
from niaaml.fitness.fitness_function import FitnessFunction

__all__ = ["MSE"]


class MSE(FitnessFunction):
    r"""Class representing the negative mean squared error as a fitness function.

    Date:
        2024

    Author:
        Laurenz Farthofer

    License:
        MIT

    Documentation:
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error

    See Also:
        * :class:`niaaml.fitness.FitnessFunction`
    """
    Name = "Mean Squared Error"

    def get_fitness(self, predicted, expected):
        r"""Return fitness value. The larger return value should represent a better fitness for the framework to work properly.

        Arguments:
            predicted (pandas.core.series.Series): Predicted values.
            expected (pandas.core.series.Series): Expected values.

        Returns:
            float: Calculated fitness value.
        """
        return  - mean_squared_error(expected, predicted)
    
    def get_bounds(self):
        #! float("-inf") leads to errors in the pipeline logic, so we use a very big number instead
        return (-1000000000.0, 0.0)
