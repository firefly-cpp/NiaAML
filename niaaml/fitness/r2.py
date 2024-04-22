from sklearn.metrics import r2_score
from niaaml.fitness.fitness_function import FitnessFunction

__all__ = ["R2"]


class R2(FitnessFunction):
    r"""Class representing the R2-score as a fitness function.

    Date:
        2024

    Author:
        Laurenz Farthofer

    License:
        MIT

    Documentation:
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score

    See Also:
        * :class:`niaaml.fitness.FitnessFunction`
    """
    Name = "R2-score"

    def get_fitness(self, predicted, expected):
        r"""Return fitness value. The larger return value should represent a better fitness for the framework to work properly.

        Arguments:
            predicted (pandas.core.series.Series): Predicted values.
            expected (pandas.core.series.Series): Expected values.

        Returns:
            float: Calculated fitness value.
        """
        return r2_score(expected, predicted)
    
    def get_bounds(self):
        #! float("-inf") leads to errors in the pipeline logic, so we use a very big number instead
        return (-100000.0, 1.0)
