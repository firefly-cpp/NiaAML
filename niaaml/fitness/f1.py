from sklearn.metrics import f1_score
from niaaml.fitness.fitness_function import FitnessFunction

__all__ = ["F1"]


class F1(FitnessFunction):
    r"""Class representing the F1-score as a fitness function.

    Date:
        2020

    Author:
        Luka Pečnik

    License:
        MIT

    Documentation:
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

    See Also:
        * :class:`niaaml.fitness.FitnessFunction`
    """
    Name = "F-score"

    def get_fitness(self, predicted, expected):
        r"""Return fitness value. The larger return value should represent a better fitness for the framework to work properly.

        Arguments:
            predicted (pandas.core.series.Series): Predicted values.
            expected (pandas.core.series.Series): Expected values.

        Returns:
            float: Calculated fitness value.
        """
        return f1_score(expected, predicted, average="weighted")
