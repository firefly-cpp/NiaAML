from niaaml.utilities import Factory
from niaaml.fitness.accuracy import Accuracy
from niaaml.fitness.cohen_kappa import CohenKappa
from niaaml.fitness.precision import Precision
from niaaml.fitness.f1 import F1
from niaaml.fitness.r2 import R2
from niaaml.fitness.mse import MSE

__all__ = ["FitnessFactory"]


class FitnessFactory(Factory):
    r"""Class with string mappings to fitness class.

    Attributes:
        _entities (Dict[str, Fitness]): Mapping from strings to fitness classes.

    See Also:
        * :class:`niaaml.utilities.Factory`
    """

    def _set_parameters(self, **kwargs):
        r"""Set the parameters/arguments of the factory."""
        self._entities = {
            "Accuracy": Accuracy,
            "Precision": Precision,
            "CohenKappa": CohenKappa,
            "F1": F1,
            "R2": R2,
            "MSE": MSE,
        }
