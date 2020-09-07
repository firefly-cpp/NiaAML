__all__ = [
    'DataReader'
]

class DataReader:
    r"""Class for implementing data readers with different sources of data.
    
	Date:
		2020

	Author
		Luka Pečnik

	License:
        MIT

	Attributes:
        X (numpy.ndarray): Array of rows from dataset without expected classification results.
        Y (numpy.array): Array of expected classification results.
    """
    X = None
    Y = None
    LabelsMapping = None

    def __init__(self, **kwargs):
        r"""Initialize data reader.
        """
        self.setParameters(**kwargs)
    
    def setParameters(self, **kwargs):
        return

    def readData(self, **kwargs):
        r"""Read data from expected source.
        """
        return self.X, self.Y
