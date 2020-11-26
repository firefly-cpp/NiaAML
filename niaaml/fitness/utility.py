from niaaml.utilities import Factory
from niaaml import fitness

__all__ = [
    'FitnessFactory'
]

class FitnessFactory(Factory):
    r"""Class with string mappings to fitness class.

    Attributes:
        _entities (Dict[str, Fitness]): Mapping from strings to fitness classes.

    See Also:
        * :class:`niaaml.utilities.Factory`
    """

    def _set_parameters(self, **kwargs):
        r"""Set the parameters/arguments of the factory.
        """
        self._entities = {
            'Accuracy': fitness.Accuracy,
            'Precision': fitness.Precision,
            'CohenKappa': fitness.CohenKappa,
            'F1': fitness.F1
        }
