import logging
import os


class Logger:
    r"""Class for logging throughout the framework.

    Date:
        2020

    Author:
        Luka Pečnik

    License:
        MIT
    """

    def __init__(self, verbose=False, output_file=None, **kwargs):
        r"""Initialize Logger.

        Arguments:
            verbose (Optional(bool)): If True, output verbose pipeline info.
            output_file (Optional(str)): If set, logger outputs content to a log file.
        """
        if output_file is not None:
            if (
                len(os.path.splitext(output_file)[1]) == 0
                or os.path.splitext(output_file)[1] != ".log"
            ):
                output_file = output_file + ".log"
        self.__logger = logging.getLogger("niaaml")
        self.__logger.setLevel(logging.INFO)

        if output_file is not None:
            fh = logging.FileHandler(output_file)
            self.__logger.addHandler(fh)

        self.__verbose = verbose

    def __del__(self):
        logging.shutdown()

    def log_progress(self, text):
        r"""Log progress message."""
        self.__logger.info(text)

    def log_pipeline(self, text):
        r"""Log pipeline info message."""
        if self.__verbose is True:
            self.__logger.info(text)

    def log_optimization_error(self, text):
        r"""Log optimization error message."""
        if self.__verbose is True:
            self.__logger.warning(text)
