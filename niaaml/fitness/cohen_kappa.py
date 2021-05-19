from sklearn.metrics import cohen_kappa_score
from niaaml.fitness.fitness_function import FitnessFunction

__all__ = ["CohenKappa"]


class CohenKappa(FitnessFunction):
    r"""Class representing the cohen's kappa as a fitness function.

    Date:
        2020

    Author:
        Luka Pečnik

    License:
        MIT

    Documentation:
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html

    See Also:
        * :class:`niaaml.fitness.FitnessFunction`
    """
    Name = "Cohen's Kappa"

    def get_fitness(self, predicted, expected):
        r"""Return fitness value. The larger return value should represent a better fitness for the framework to work properly.

        Arguments:
            predicted (pandas.core.series.Series): Predicted values.
            expected (pandas.core.series.Series): Expected values.

        Returns:
            float: Calculated fitness value.
        """
        return cohen_kappa_score(expected, predicted)
