__all__ = ["PipelineComponent"]


class PipelineComponent:
    r"""Class for implementing pipeline components.

    Date:
        2020

    Author:
        Luka Pečnik

    License:
        MIT

    Attributes:
        Name (str): Name of the pipeline component.
        _params (Dict[str, ParameterDefinition]): Dictionary of components's parameters with possible values. Possible parameter values are given as an instance of the ParameterDefinition class.

    See Also:
        * :class:`niaaml.utilities.ParameterDefinition`
    """
    Name = None

    def __init__(self, **kwargs):
        r"""Initialize pipeline component.

        Notes:
            _params variable should not be static as in some cases it is instance specific. See * :class:`niaaml.preprocessing.feature_selection.select_k_best.SelectKBest` for example.
        """
        self._params = dict()
        self.set_parameters(**kwargs)

    def set_parameters(self, **kwargs):
        r"""Set the parameters/arguments of the pipeline component."""
        return

    def get_params_dict(self):
        r"""Return parameters definition dictionary."""
        return self._params

    def to_string(self):
        r"""User friendly representation of the object.

        Returns:
            str: User friendly representation of the object.
        """
        return "Name: {name}\nArguments:\n{args}"

    def _parameters_to_string(self, dictionary):
        r"""User friendly representation of component's parameters.

        Arguments:
            dictionary (dict): Dictionary of parameters.

        Returns:
            str: User friendly representation of component's parameters.
        """
        args_string = ""
        for key in dictionary:
            args_string += "\t" + key + " = " + str(dictionary[key]) + "\n"
        if len(args_string) == 0:
            args_string = "None"
        return args_string
