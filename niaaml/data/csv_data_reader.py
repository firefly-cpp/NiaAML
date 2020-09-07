import csv
import numpy as np
from niaaml.data.data_reader import DataReader
from niaaml.utility import encodeLabels

__all__ = ['CSVDataReader']

class CSVDataReader(DataReader):
    def setParameters(self, src, **kwargs):
        DataReader.setParameters(self, **kwargs)
        self.src = src

    def readData(self, **kwargs):
        X = []
        Y = []
        with open(self.src) as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)
            for row in reader:
                X.append(row[1:])
                Y.append(row[0])

            self.X = np.array(X).astype(np.float)

            self.LabelsMapping = encodeLabels(Y)
            self.Y = np.array([self.LabelsMapping[y] for y in Y]).astype(np.uintc)